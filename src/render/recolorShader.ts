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
out vec2 v_uv;

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
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_video;   // кадр камеры (полное разрешение)
uniform sampler2D u_mask;    // маска волос 256x256 (билинейный апскейл бесплатно)
uniform vec3  u_targetColor; // целевой цвет волос, RGB 0..1
uniform float u_strength;    // сила окраски 0..1 (Intense/Pastel)
uniform float u_satScale;    // множитель насыщенности (Pastel приглушает)
uniform float u_mirror;      // 1.0 = отразить по X (фронталка)
uniform vec2  u_maskTexel;   // размер тексела маски (1/ширина, 1/высота)
uniform float u_feather;     // радиус размытия краёв маски в текселях (этап 2)
uniform float u_split;       // split-view: x-позиция (0..1) или <0 если выключен
uniform float u_lumaShift;   // сдвиг яркости к тону цели (0 — сохранять L, 1 — сильно)
uniform sampler2D u_lut;     // LUT-атлас: x=яркость волоса, y=оттенок
uniform float u_lutRow;      // v-координата строки оттенка (0..1); <0 — использовать HSL

// ---- RGB <-> HSL (стандартные) ----
vec3 rgb2hsl(vec3 c) {
  float maxc = max(max(c.r, c.g), c.b);
  float minc = min(min(c.r, c.g), c.b);
  float l = (maxc + minc) * 0.5;
  float h = 0.0;
  float s = 0.0;
  float d = maxc - minc;
  if (d > 1e-5) {
    s = l > 0.5 ? d / (2.0 - maxc - minc) : d / (maxc + minc);
    if (maxc == c.r)      h = (c.g - c.b) / d + (c.g < c.b ? 6.0 : 0.0);
    else if (maxc == c.g) h = (c.b - c.r) / d + 2.0;
    else                  h = (c.r - c.g) / d + 4.0;
    h /= 6.0;
  }
  return vec3(h, s, l);
}

float hue2rgb(float p, float q, float t) {
  if (t < 0.0) t += 1.0;
  if (t > 1.0) t -= 1.0;
  if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
  if (t < 1.0 / 2.0) return q;
  if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
  return p;
}

vec3 hsl2rgb(vec3 hsl) {
  float h = hsl.x, s = hsl.y, l = hsl.z;
  if (s < 1e-5) return vec3(l);
  float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
  float p = 2.0 * l - q;
  return vec3(
    hue2rgb(p, q, h + 1.0 / 3.0),
    hue2rgb(p, q, h),
    hue2rgb(p, q, h - 1.0 / 3.0)
  );
}

// Guided feathering (этап 2, вариант «лучше» из брифа): joint-bilateral —
// уточняем низкоразрешённую (256) маску по ПОЛНОразрешённому видеокадру как гайду.
// Каждый сосед взвешиваем (а) пространственно и (б) по СХОДСТВУ ЦВЕТА с центром.
// Так край маски «прилипает» к границе волос/кожи и идёт вдоль прядей, а не
// обрезан по грубой сетке 256. Один проход, без доп. фреймбуферов.
float guidedMask(vec2 uv, vec3 centerColor) {
  if (u_feather <= 0.01) return texture(u_mask, uv).r;
  vec2 step = u_maskTexel * u_feather * 1.4; // шире шаг — 3x3 покрывает прежнюю зону
  float sum = 0.0;
  float wsum = 0.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 o = vec2(float(x), float(y)) * step;
      float m = texture(u_mask, uv + o).r;
      vec3 c = texture(u_video, uv + o).rgb;
      vec3 d = c - centerColor;
      float wColor = exp(-dot(d, d) * 22.0);          // близкий цвет -> больше вес
      float wSpace = exp(-float(x * x + y * y) * 0.5);  // ближе -> больше вес
      float w = wColor * wSpace;
      sum += m * w;
      wsum += w;
    }
  }
  return wsum > 0.0 ? sum / wsum : texture(u_mask, uv).r;
}

void main() {
  vec2 uv = v_uv;
  // Зеркалим фронтальную камеру по горизонтали.
  vec2 sampleUv = vec2(mix(uv.x, 1.0 - uv.x, u_mirror), uv.y);

  vec3 orig = texture(u_video, sampleUv).rgb;
  // Guided feathering + лёгкий ремап края внутрь, чтобы цвет не выползал на кожу.
  float mask = smoothstep(0.4, 0.82, guidedMask(sampleUv, orig));

  vec3 recolored;
  if (u_lutRow >= 0.0) {
    // LUT-подход (как в референсе): по яркости волоса берём цвет из рампы оттенка,
    // построенной из реального фото пряди (тень->средний тон->блик).
    // Яркость волоса растягиваем на весь диапазон рампы (типичные волосы тёмные —
    // иначе сэмплим только мутный тёмный край и оттенок не читается).
    float lum = dot(orig, vec3(0.299, 0.587, 0.114));
    // гамма поднимает тёмные волосы в средне-светлую часть рампы — так читается
    // характерный цвет оттенка, а не его мутный тёмный край.
    float t = clamp(pow(clamp(lum, 0.0, 1.0), 0.5), 0.0, 1.0);
    recolored = texture(u_lut, vec2(t, u_lutRow)).rgb;
  } else {
    // HSL: целевой H/S, L из оригинала со сдвигом к тону цели.
    vec3 tgtHsl = rgb2hsl(u_targetColor);
    vec3 origHsl = rgb2hsl(orig);
    float newL = clamp(origHsl.z + (tgtHsl.z - 0.5) * u_lumaShift, 0.0, 1.0);
    // Светлые оттенки (блонд/осветлители) приглушаем по насыщенности — иначе
    // получается жёлтый, а не мягкий блонд.
    float lightAttn = smoothstep(0.58, 0.95, tgtHsl.z);
    float sat = tgtHsl.y * u_satScale * (1.0 - lightAttn * 0.72);
    // В бликах волос (яркие пиксели) снижаем насыщенность — мягкое сияние вместо
    // резкого пересвеченного цвета («меньше граней света»).
    sat *= mix(1.0, 0.45, smoothstep(0.55, 0.92, origHsl.z));
    vec3 recHsl = vec3(tgtHsl.x, sat, newL);
    recolored = hsl2rgb(recHsl);
  }

  // Смешиваем по маске и силе. Вне маски (mask=0) — оригинал нетронут.
  float a = mask * u_strength;

  // Split-view (before/after): слева от линии — оригинал, справа — фильтр.
  // Используем экранный uv.x (не зеркалим), чтобы лево всегда было слева.
  if (u_split >= 0.0 && uv.x < u_split) {
    a = 0.0;
  }

  vec3 result = mix(orig, recolored, a);

  // Тонкая вертикальная линия-разделитель по центру.
  if (u_split >= 0.0 && abs(uv.x - u_split) < 0.0016) {
    result = vec3(1.0);
  }

  fragColor = vec4(result, 1.0);
}
`;
