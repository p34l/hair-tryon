/**
 * GLSL-шейдеры для перекрашивания волос (WebGL2 / GLSL ES 3.00).
 *
 * Идея фрагментного шейдера (как в Google "Real-time Hair Recoloring",
 * arXiv 1907.06740): берём H и S целевого цвета, но СОХРАНЯЕМ L (яркость)
 * исходного пикселя — так остаются тени, блики и текстура прядей, а не
 * плоская заливка. Смешиваем по значению маски и параметру strength.
 *
 * rgb<->hsl — стандартные функции (референс: The Book of Shaders / LYGIA).
 */

export const VERTEX_SHADER = /* glsl */ `#version 300 es
precision highp float;

// Полноэкранный треугольник: позиции генерим из gl_VertexID, без буфера атрибутов.
out highp vec2 v_uv;

void main() {
  // Три вершины, покрывающие весь клип-спейс.
  vec2 pos = vec2(
    (gl_VertexID == 2) ? 3.0 : -1.0,
    (gl_VertexID == 1) ? 3.0 : -1.0
  );
  // Y инвертируем: текстуры (video/маска) грузятся строкой 0 сверху, а в клип-
  // спейсе y=+1 — верх экрана. Без флипа картинка выводится вверх ногами.
  v_uv = vec2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
  gl_Position = vec4(pos, 0.0, 1.0);
}
`;

export const FRAGMENT_SHADER = /* glsl */ `#version 300 es
// По умолчанию mediump (быстрее на мобильных GPU); координаты текстур — highp
// точечно (иначе на high-res дрожит сэмплинг).
precision mediump float;

in highp vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_video;   // кадр камеры (полное разрешение)
uniform sampler2D u_mask;    // маска волос 256x256 (билинейный апскейл бесплатно)
uniform vec2  u_coverScale;  // object-fit: cover в шейдере (масштаб UV вокруг центра)
uniform vec3  u_targetLab;   // целевой цвет в OKLab (посчитан на CPU из линейного sRGB)
uniform float u_strength;    // сила окраски 0..1 (Intense/Pastel)
uniform float u_satScale;    // множитель насыщенности (Pastel приглушает)
uniform float u_mirror;      // 1.0 = отразить по X (фронталка)
uniform vec2  u_maskTexel;   // размер тексела маски (1/ширина, 1/высота)
uniform float u_feather;     // радиус размытия краёв маски в текселях (этап 2)
uniform float u_split;       // split-view: x-позиция (0..1) или <0 если выключен
uniform float u_lumaShift;   // сдвиг яркости к тону цели (0 — сохранять L, 1 — сильно)
uniform sampler2D u_lut;     // LUT-атлас: x=яркость волоса, y=оттенок
uniform float u_lutRow;      // v-координата строки оттенка (0..1); <0 — использовать HSL
uniform float u_colorSharp;  // резкость веса по цвету в joint-bilateral (этап 3)
uniform float u_edgeLow;     // нижний край финального smoothstep-ремапа маски
uniform float u_edgeHigh;    // верхний край финального smoothstep-ремапа маски

// ---- sRGB <-> linear ----
// Видео-текстура — обычный RGBA8 (sRGB-байты), поэтому srgb2lin применяем в шейдере
// к сэмплу видео (orig) и к LUT; lin2srgb — на выходе в дефолтный фреймбуфер.
vec3 srgb2lin(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 lin2srgb(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }

// ---- linear sRGB <-> OKLab (Björn Ottosson) ----
// Перцептивно-равномерное пространство: L = воспринимаемая яркость, (a,b) = цвет.
// Берём L (со всей текстурой прядей) из оригинала, а (a,b) — от цели; переходы
// тон-в-тон выходят ровными, без «грязи» HSL на тёмных/ярких участках.
vec3 linToOklab(vec3 c) {
  float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
  float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
  float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
  float l_ = pow(max(l, 0.0), 1.0 / 3.0);
  float m_ = pow(max(m, 0.0), 1.0 / 3.0);
  float s_ = pow(max(s, 0.0), 1.0 / 3.0);
  return vec3(
    0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
  );
}
vec3 oklabToLin(vec3 lab) {
  float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  float l = l_ * l_ * l_;
  float m = m_ * m_ * m_;
  float s = s_ * s_ * s_;
  return vec3(
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
  );
}

// Joint-bilateral matting-апсемпл (этап 3): уточняем низкоразрешённую (256)
// МЯГКУЮ маску по ПОЛНОразрешённому видеокадру как гайду. Каждый сосед
// взвешиваем (а) пространственно и (б) по СХОДСТВУ ЦВЕТА с центром. С мягкой
// (confidence) маской и резким цветовым весом край «прилипает» к реальному
// переходу волосы→кожа и идёт вдоль прядей: hair-цветные соседи тянут маску
// вверх на волосах, кожа-цветные исключаются на коже. 5x5 для плавности края,
// один проход без доп. фреймбуферов.
// 5x5 joint-bilateral — мягкий край вдоль прядей; вес по цвету отсекает кожу.
float guidedMask(highp vec2 uv, vec3 centerColor) {
  if (u_feather <= 0.01) return texture(u_mask, uv).r;
  highp vec2 step = u_maskTexel * u_feather;
  float sum = 0.0;
  float wsum = 0.0;
  for (int y = -2; y <= 2; y++) {
    for (int x = -2; x <= 2; x++) {
      highp vec2 o = vec2(float(x), float(y)) * step;
      float m = texture(u_mask, uv + o).r;
      vec3 c = texture(u_video, uv + o).rgb;
      vec3 d = c - centerColor;
      // Резкий вес по цвету: граница тянется по реальному переходу, кожа отсекается.
      float wColor = exp(-dot(d, d) * u_colorSharp);
      float wSpace = exp(-float(x * x + y * y) * 0.30); // ближе -> больше вес
      float w = wColor * wSpace;
      sum += m * w;
      wsum += w;
    }
  }
  return wsum > 0.0 ? sum / wsum : texture(u_mask, uv).r;
}

void main() {
  highp vec2 uv = v_uv;
  // Зеркалим фронтальную камеру по горизонтали, затем object-fit: cover в UV
  // (рендерим только видимый кроп — буфер уже в аспекте сцены, не видео).
  highp vec2 sampleUv = vec2(mix(uv.x, 1.0 - uv.x, u_mirror), uv.y);
  sampleUv = (sampleUv - 0.5) * u_coverScale + 0.5;

  // Видео-текстура теперь RGBA8 (sRGB-байты), поэтому декодируем в linear вручную.
  vec3 origSrgb = texture(u_video, sampleUv).rgb;
  // Matting-апсемпл + ремап края: центр волос→1, кожа→0, граница≈0.5. Сравнение
  // цвета в guidedMask — в sRGB (соседи тоже сэмплятся как sRGB), это ок для меры
  // близости. В линейный переводим уже результат (orig) для перекраски/микса.
  float mask = smoothstep(u_edgeLow, u_edgeHigh, guidedMask(sampleUv, origSrgb));
  vec3 orig = srgb2lin(origSrgb);

  // orig — уже линейный (SRGB8-текстура). Перекраска тоже в линейном.
  vec3 recolored;
  if (u_lutRow >= 0.0) {
    // LUT-подход (как в референсе): по яркости волоса берём цвет из рампы оттенка.
    float lum = dot(orig, vec3(0.2126, 0.7152, 0.0722)); // линейная luma
    float t = clamp(pow(clamp(lum, 0.0, 1.0), 0.5), 0.0, 1.0);
    recolored = srgb2lin(texture(u_lut, vec2(t, u_lutRow)).rgb); // LUT в sRGB -> lin
  } else {
    // OKLab: L (и текстуру прядей) берём из оригинала со сдвигом к L цели; цвет
    // (a,b) — направление цели с нужной хромой. OKLab цели посчитан на CPU.
    vec3 oLab = linToOklab(orig);
    vec3 tLab = u_targetLab;
    float tC = length(tLab.yz);
    vec2 dir = tC > 1e-4 ? tLab.yz / tC : vec2(0.0);
    // L из оригинала, смещённый к тону цели — сохраняем тени/блики/пряди.
    float newL = clamp(oLab.x + (tLab.x - 0.5) * u_lumaShift, 0.0, 1.0);
    // Хрома цели; Pastel приглушает (u_satScale). На очень тёмных волосах слегка
    // поднимаем, иначе тёмные пряди выходят почти серыми.
    float chroma = tC * u_satScale;
    float darkLift = 1.0 - smoothstep(0.0, 0.35, oLab.x);
    chroma *= 1.0 + 0.35 * darkLift;
    // В ярких бликах чуть снижаем хрому — мягкое сияние без неонового пересвета.
    chroma *= mix(1.0, 0.7, smoothstep(0.78, 1.0, oLab.x));
    recolored = oklabToLin(vec3(newL, dir * chroma));
  }

  // Смешиваем по маске и силе. Вне маски (mask=0) — оригинал нетронут.
  float a = mask * u_strength;

  // Split-view (before/after): слева от линии — оригинал. Сама линия и ручка
  // рисуются DOM-оверлеем (чёткие, с зоной захвата) — здесь только маскируем.
  // Используем экранный uv.x (не зеркалим), чтобы лево всегда было слева.
  if (u_split >= 0.0 && uv.x < u_split) {
    a = 0.0;
  }

  // Микс в линейном свете, на выходе кодируем в sRGB (дефолтный фреймбуфер).
  vec3 result = lin2srgb(mix(orig, recolored, a));

  // Лёгкий дизеринг разбивает бандинг на градиенте перекраски (8-битный вывод).
  float dither = (fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0;
  result += dither;

  fragColor = vec4(result, 1.0);
}
`;
