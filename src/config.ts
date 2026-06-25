/**
 * Глобальные константы пайплайна. Всё, что нужно подбирать визуально
 * (alpha EMA, разрешение инференса, радиус feathering), живёт здесь —
 * чтобы не искать магические числа по файлам.
 */

/** Разрешение, на котором гоняем сегментацию. Бриф: строго 256x256. */
export const INFERENCE_SIZE = 256;

/** Индексы классов модели SelfieMulticlass. */
export const SEG_CLASS = {
  background: 0,
  hair: 1,
  bodySkin: 2,
  faceSkin: 3,
  clothes: 4,
  others: 5,
} as const;

/**
 * Пути к моделям. Лежат в public/models/ (см. этап 0).
 * WASM-рантайм MediaPipe берём с CDN jsdelivr.
 */
export const MODEL = {
  segmenter: '/models/selfie_multiclass_256x256.tflite',
  // Выделенная модель сегментации волос (2 класса: bg/hair). Её GPU-делегат
  // намного устойчивее мультикласса: на части Android мультиклас на GPU отдаёт
  // 0 масок ("mask count 0 vs 6"), а эта модель на GPU работает. Используется
  // как GPU-фолбэк, чтобы НЕ падать на медленный CPU. Тот же hair-класс = 1.
  segmenterHair: '/models/hair_segmenter.tflite',
  faceLandmarker: '/models/face_landmarker.task',
  // WASM-рантайм с CDN, версия жёстко привязана к запиненной в package.json
  // (0.10.35). Совпадение версий обязательно — иначе "ModuleFactory not set".
  // Локально из /public нельзя: MediaPipe грузит loader через dynamic import(),
  // а Vite не отдаёт /public-файлы как модули в dev.
  // ВАЖНО: версия WASM ДОЛЖНА совпадать с npm-версией @mediapipe/tasks-vision,
  // иначе "ModuleFactory not set". 0.10.36-rc iOS-утечку НЕ исправил — вернулись на 0.10.35.
  wasmRoot: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm',
} as const;

/**
 * Адаптивный темпоральный EMA (этап 2, motion-compensated).
 * alpha не фиксирован: считаем движение как среднюю нормированную |cur−prev|
 * по всей маске (0..1) и интерполируем
 *   alpha = clamp(EMA_ALPHA_STILL + motion*EMA_MOTION_GAIN, EMA_ALPHA_STILL, MASK_EMA_ALPHA)
 * В покое alpha мал (≈0.22) → край не «кипит»; при быстром движении растёт почти
 * до 1.0 → маска «клацает» на свежую и не отстаёт от волос (хвост ~5%).
 */
export const MASK_EMA_ALPHA = 0.95;   // верхняя граница (почти без хвоста при быстром движении)
export const EMA_ALPHA_STILL = 0.22;  // нижняя граница (стабильность в покое)
export const EMA_MOTION_GAIN = 16.0;  // насколько резко alpha растёт с движением

/**
 * Framerate-компенсация EMA. alpha выше подобран под опорную частоту обновления
 * маски ~30 Гц. Если маски приходят реже (Android-CPU-фолбэк: ~12–15 Гц), то же
 * alpha даёт ВДВОЕ больше лага в реальном времени (EMA сглаживает «по кадрам», а
 * не по времени). Компенсируем по фактическому интервалу dt между масками:
 *   alpha_eff = 1 − (1 − alpha)^(dt / EMA_REF_DT_MS)
 * При 30 Гц (dt≈33мс) alpha не меняется (десктоп/iOS не затронуты); при падении
 * частоты alpha растёт, удерживая постоянную постоянную времени в миллисекундах.
 */
export const EMA_REF_DT_MS = 33.3;    // опорный интервал (≈30 Гц), при нём alpha без изменений
export const EMA_DT_RATIO_MAX = 4.0;  // потолок dt/ref — чтобы пауза/столл не «защёлкивал» alpha до 1

/**
 * Минимальный интервал между отправками кадра в воркер сегментации (мс).
 * Coalescing (latest-wins) держит маску свежей, но БЕЗ кепа воркер на мобильном
 * GPU молотит на 100% и отнимает кадры у рендера — особенно при движении головы.
 * ~22 Гц сегментации — маска хорошо «держится» на голове при движении.
 */
export const SEG_MIN_INTERVAL_MS = 45;

/**
 * Включать ли второй MediaPipe-граф (FaceLandmarker) для exclusion-маски бороды.
 * На слабком мобильном GPU это ВТОРАЯ тяжёлая модель, которая делит тот же GPU с
 * сегментатором и рендером — на Android сегментатор уже сам пожирает ~600мс/с GPU
 * (инференс 100мс × 6 Гц), и рендер падает до ~7 FPS. Борода — necessity низкого
 * приоритета, поэтому по умолчанию ВЫКЛ: освобождаем GPU под рендер.
 */
export const ENABLE_BEARD_EXCLUSION = false;

/**
 * Интервал детекта FaceLandmarker (мс). При быстром движении MediaPipe теряет
 * трекинг и гоняет тяжёлый полный детектор на главном потоке — реже = меньше
 * просадок FPS в движении. Борода двигается медленно, ~4 Гц хватает.
 */
export const FACE_DETECT_INTERVAL_MS = 250;

/** Радиус размытия краёв маски в текселях маски. Больше = мягче край (feather). */
export const FEATHER_RADIUS = 2.6;

/**
 * Потолок высоты рендер-таргета в физ. пикселях (этап 5). Канвас держим в
 * аспекте видео (иначе sampleUv растянет картинку), но масштабируем под
 * фактический размер вывода × devicePixelRatio с потолком — чтобы шейдер не
 * считал per-pixel на 1080p зря. 1280 — чёткая картинка на high-DPI экранах.
 */
export const MAX_RENDER_HEIGHT = 768;

/**
 * Matting-край (этап 3). Joint-bilateral апсемпл мягкой маски по hi-res видео.
 * GUIDED_COLOR_SHARP — резкость веса по ЦВЕТУ: чем больше, тем сильнее край
 * прилипает к реальной границе волосы↔кожа (кожа исключается жёстче).
 * MASK_EDGE_LOW/HIGH — финальный smoothstep-ремап уточнённой маски: центр волос
 * → 1.0, кожа → 0.0, граница ≈0.5. Сдвинут чуть выше 0.5, чтобы цвет не заезжал
 * на кожу/лоб, но мягкость прядей сохранялась.
 */
export const GUIDED_COLOR_SHARP = 42.0;
// Широкий мягкий ремап края: фейд альфы идёт по более широкому диапазону маски,
// поэтому граница перекраски не «обрезана», а плавно растворяется (особенно
// заметно на ярких цветах). Ядро волос (>HIGH) красится полностью.
export const MASK_EDGE_LOW = 0.38;
export const MASK_EDGE_HIGH = 0.90;

/**
 * Сдвиг яркости волос к тону выбранного цвета (0 — строго сохранять яркость
 * оригинала, 1 — сильно). Позволяет тёмным оттенкам затемнять, а светлым —
 * осветлять волосы, сохраняя текстуру прядей.
 */
// Чем меньше — тем больше сохраняется родная светотень/текстура волос (тени между
// прядями, объём), и цвет «осаждается» как настоящая краска, а не плоская заливка.
// Высокие значения рaвномерно поднимают яркость → «вылитый»/плоский вид.
export const LUMA_SHIFT = 0.45;

/** Сколько секунд отсчёта перед снимком (кнопка фото). */
export const CAPTURE_COUNTDOWN = 3;

/**
 * Логотип бренда — локальный public/logo.png (силуэт с убранным белым фоном).
 * Один файл и для анимации загрузки, и для брендинга/дисклеймера.
 */
export const LOGO = {
  icon: '/logo.png',
  loading: '/loading.gif', // реальная анимация загрузки из референса (скачана локально)
} as const;

/**
 * Параметры режимов интенсивности.
 * Pastel — низкая сила + приглушённая насыщенность; Intense — наоборот.
 */
export const INTENSITY_PARAMS = {
  intense: { strength: 0.8, satScale: 0.9 },
  // Для Pastel цвет берётся из отдельного pastelHex (он уже светлый/приглушённый).
  pastel: { strength: 0.62, satScale: 0.85 },
} as const;

import type { ColorPreset } from './types';

/**
 * Полный реальный каталог Schwarzkopf LIVE (77 оттенков). Источник:
 *  - i18n-ответ референса (api.ar.schwarzkopf.international/v1/client/i18n) —
 *    name, claim, id(GUID), sliderUrl (фото пряди), productUrl (фото упаковки);
 *  - цвет hex — реальный из AR-API (.../v1/client/products/{guid}, поле rgbColor).
 * image — фото упаковки, swatchImage — фото пряди (hotlink на ассеты sk-qr.com).
 */
const RAW_PRESETS: ColorPreset[] = [
  // Реальный каталог Schwarzkopf LIVE (77). hex — цвет оттенка из AR-API
  // (насыщенный, читается на волосах). swatchImage — фото пряди, image — упаковка.
  { id: "00P_0", code: "00P", name: "Bold Blonde", subtitle: "Anti-Brassiness | Up to 80% less hair breakage", hex: "#f2ca8c", image: "https://sk-qr.com/assets/images/appearance/LIVE/00P_PowderBleachBoldBlonde.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/00P_PowderBleachBoldBlonde.jpg", real: true },
  { id: "030_1", code: "030", name: "MangoTwist", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#a84108", image: "https://sk-qr.com/assets/images/appearance/LIVE/030_MangoTwist.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/030_MangoTwist.jpg", real: true },
  { id: "059_2", code: "059", name: "Blue Berry", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#171e5a", image: "https://sk-qr.com/assets/images/appearance/LIVE/059.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/059_MA_LIVE_COLO_FB_Base_XXXX_LE22_Ep2_0523_.jpg", real: true },
  { id: "087_3", code: "087", name: "Mystic Violet", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#370c32", image: "https://sk-qr.com/assets/images/appearance/LIVE/087_MysticViolet.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/087_MysticViolet.jpg", real: true },
  { id: "090_4", code: "090", name: "Cosmic Blue", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#1f2336", image: "https://sk-qr.com/assets/images/appearance/LIVE/090.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/090.jpg", real: true },
  { id: "077_5", code: "077", name: "Caramel Copper", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#a63a04", image: "https://sk-qr.com/assets/images/appearance/LIVE/077.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/077_CaramelCopper.jpg", real: true },
  { id: "880_6", code: "880", name: "Espresso Brown", subtitle: "100% Grey Coverage | Anti-Fading Effect", hex: "#20120a", image: "https://sk-qr.com/assets/images/appearance/LIVE/880%20espresso%20brown.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/880%20espresso%20brown.jpg", real: true },
  { id: "00B_7", code: "00B", name: "Ice Blonde", subtitle: "Up to 4 Shades lighter | Anti-Yellow Effect", hex: "#ffe1aa", image: "https://sk-qr.com/assets/images/appearance/LIVE/00B%20ice%20blonde.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/00B%20ice%20blonde.jpg", real: true },
  { id: "046_8", code: "046", name: "Cyber Purple", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#571442", image: "https://sk-qr.com/assets/images/appearance/LIVE/046_CyberPurple.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/046_CyberPurple.jpg", real: true },
  { id: "00A_9", code: "00A", name: "Absolute Platinum", subtitle: "Anti-Brassiness | Up to 80% less hair breakage", hex: "#f0dea8", image: "https://sk-qr.com/assets/images/appearance/LIVE/00A_IntenseLightenerAbsolutePlatinum.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/00A_IntenseLightenerAbsolutePlatinum.jpg", real: true },
  { id: "14_10", code: "1.4", name: "Blueberry Black", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#101823", image: "https://sk-qr.com/assets/images/appearance/LIVE/1.4.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/1.4.jpg", real: true },
  { id: "880_11", code: "880", name: "Tempting Chocolate", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#24120a", image: "https://sk-qr.com/assets/images/appearance/LIVE/880%20tempting%20chocolate.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/880%20tempting%20chocolate.jpg", real: true },
  { id: "043_12", code: "043", name: "Red Passion", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#841e2a", image: "https://sk-qr.com/assets/images/appearance/LIVE/043_RedPassion.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/043_RedPassion.jpg", real: true },
  { id: "086_13", code: "086", name: "Pure Purple", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#781f3b", image: "https://sk-qr.com/assets/images/appearance/LIVE/086_PurePurple.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/086_PurePurple.jpg", real: true },
  { id: "091_14", code: "091", name: "Raspberry Rebel", subtitle: "Intense or Pastel look | Up to 15 Washes of Colour Vibrancy", hex: "#a5195a", image: "https://sk-qr.com/assets/images/appearance/LIVE/091.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/091.jpg", real: true },
  { id: "089_15", code: "089", name: "Bitter Sweet Chocolate", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#23140a", image: "https://sk-qr.com/assets/images/appearance/LIVE/089.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/089.jpg", real: true },
  { id: "092_16", code: "092", name: "Pillar Box Red", subtitle: "Intense or Pastel look | Up to 15 Washes of Colour Vibrancy", hex: "#a80a1e", image: "https://sk-qr.com/assets/images/appearance/LIVE/092.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/092.jpg", real: true },
  { id: "044_17", code: "044", name: "Berry Red", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#5a0523", image: "https://sk-qr.com/assets/images/appearance/LIVE/044.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/044_MA_LIVE_COLO_FB_Base_XXXX_LE22_Ep2_0523_.jpg", real: true },
  { id: "099_18", code: "099", name: "Pitch Black", subtitle: "100% Grey Coverage | Anti-Fading Effect", hex: "#17171a", image: "https://sk-qr.com/assets/images/appearance/LIVE/099%20Pitch%20black.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/099%20Pitch%20black.jpg", real: true },
  { id: "M05_19", code: "M05", name: "Truffle Temptation", subtitle: "Intense care with oil | Leaves hair healthy & supple", hex: "#371d0f", image: "https://sk-qr.com/assets/images/appearance/LIVE/M05.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/M05.jpg", real: true },
  { id: "10_20", code: "1.0", name: "Raven Black", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#282323", image: "https://sk-qr.com/assets/images/appearance/LIVE/1.0%20raven%20black.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/1.0%20raven%20black.jpg", real: true },
  { id: "088_21", code: "088", name: "Urban Brown", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#582b12", image: "https://sk-qr.com/assets/images/appearance/LIVE/088_UrbanBrown.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/088_UrbanBrown.jpg", real: true },
  { id: "095_22", code: "095", name: "Electric Blue", subtitle: "Intense or Pastel look | Up to 15 Washes of Colour Vibrancy", hex: "#1b4999", image: "https://sk-qr.com/assets/images/appearance/LIVE/095%20Electric%20Blue.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/095%20Electric%20Blue.jpg", real: true },
  { id: "77_23", code: "7.7", name: "Bright Cinnamon", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#9e4b0f", image: "https://sk-qr.com/assets/images/appearance/LIVE/7.7.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/7.7.jpg", real: true },
  { id: "P121_24", code: "P121", name: "Denim Steel", subtitle: "Creates soft pastel hue | Up to 8 washes", hex: "#5fbeff", image: "https://sk-qr.com/assets/images/appearance/LIVE/P121.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/P121.jpg", real: true },
  { id: "B11_25", code: "B11", name: "Frosty Blonde", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#ebd7c0", image: "https://sk-qr.com/assets/images/appearance/LIVE/B11.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/B11.jpg", real: true },
  { id: "U69_26", code: "U69", name: "Amethyst Chrome", subtitle: "Cool tones with Metallic Shine | Anti-Fading Effect", hex: "#702d5c", image: "https://sk-qr.com/assets/images/appearance/LIVE/U69.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/U69.jpg", real: true },
  { id: "40_27", code: "4.0", name: "Dark Mocca", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#3e261c", image: "https://sk-qr.com/assets/images/appearance/LIVE/4.0.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/4.0.jpg", real: true },
  { id: "00B_28", code: "00B", name: "Max Blonde", subtitle: "Anti-Brassiness | Up to 80% less hair breakage", hex: "#ffe6aa", image: "https://sk-qr.com/assets/images/appearance/LIVE/00B%20Max%20Blond.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/00B%20Max%20Blond.jpg", real: true },
  { id: "1021_29", code: "10.21", name: "Baby Blond", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#e2d2b6", image: "https://sk-qr.com/assets/images/appearance/LIVE/10.21.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/10.21.jpg", real: true },
  { id: "L61_30", code: "L61", name: "Nude Bronde", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#856d51", image: "https://sk-qr.com/assets/images/appearance/LIVE/l61-nude-bronde.png", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/l61-nude-bronde.jpg", real: true },
  { id: "U71_31", code: "U71", name: "Silver Chrome", subtitle: "Cool tones with Metallic Shine | Anti-Fading Effect", hex: "#c5c7c5", image: "https://sk-qr.com/assets/images/appearance/LIVE/U71%20silver%20chrome.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/U71%20silver%20chrome.jpg", real: true },
  { id: "B13_32", code: "B13", name: "Pearl Blonde", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#eedcb6", image: "https://sk-qr.com/assets/images/appearance/LIVE/B13.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/B13.jpg", real: true },
  { id: "U71_33", code: "U71", name: "Metallic Silver", subtitle: "Cool tones with Metallic Shine | Anti-Fading Effect", hex: "#d2dcde", image: "https://sk-qr.com/assets/images/appearance/LIVE/U71%20metallic%20silver.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/U71%20metallic%20silver.jpg", real: true },
  { id: "P123_34", code: "P123", name: "Rose Gold", subtitle: "Creates soft pastel hue | Up to 8 washes", hex: "#fcafc0", image: "https://sk-qr.com/assets/images/appearance/LIVE/P123.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/P123.jpg", real: true },
  { id: "U67_35", code: "U67", name: "Blue Mercury", subtitle: "Cool tones with Metallic Shine | Anti-Fading Effect", hex: "#213c74", image: "https://sk-qr.com/assets/images/appearance/LIVE/U67.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/U67.jpg", real: true },
  { id: "040_36", code: "040", name: "Cinnamon Cookie Butter", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#582f0f", image: "https://sk-qr.com/assets/images/appearance/LIVE/040-cinnamon-cookie-butter.png", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/040-cinnamon-cookie-butter.jpg", real: true },
  { id: "757_37", code: "7.57", name: "Sweet Toffee", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#9b501c", image: "https://sk-qr.com/assets/images/appearance/LIVE/757.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/757.jpg", real: true },
  { id: "L74_38", code: "L74", name: "Tangerine Twist", subtitle: "2 in 1 Lightening and Colour | Anti-Fading Effect for vibrant results", hex: "#af400c", image: "https://sk-qr.com/assets/images/appearance/LIVE/L74.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/L74.jpg", real: true },
  { id: "105_39", code: "105", name: "Mauve Kiss", subtitle: "2 in 1 Lightening + Pastel | Long-lasting Pastel Colour", hex: "#b87389", image: "https://sk-qr.com/assets/images/appearance/LIVE/105.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/105.jpg", real: true },
  { id: "101_40", code: "101", name: "Cool Rose", subtitle: "2 in 1 Lightening + Pastel | Long-lasting Pastel Colour", hex: "#e7b6b0", image: "https://sk-qr.com/assets/images/appearance/LIVE/101.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/101.jpg", real: true },
  { id: "P120_41", code: "P120", name: "Lilac Crush", subtitle: "Creates soft pastel hue | Up to 8 washes", hex: "#b48cf0", image: "https://sk-qr.com/assets/images/appearance/LIVE/P120.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/P120.jpg", real: true },
  { id: "085_42", code: "085", name: "Vibrant Orange", subtitle: "Intense or Pastel look | Up to 15 Washes of Colour Vibrancy", hex: "#a52a06", image: "https://sk-qr.com/assets/images/appearance/LIVE/085.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/085_VibrantOrange.jpg", real: true },
  { id: "M08_43", code: "M08", name: "Cranberry Bliss", subtitle: "Intense care with oil | Leaves hair healthy & supple", hex: "#5a0510", image: "https://sk-qr.com/assets/images/appearance/LIVE/M08.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/M08.jpg", real: true },
  { id: "890_44", code: "890", name: "Espresso Martini", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#1f1400", image: "https://sk-qr.com/assets/images/appearance/LIVE/890-espresso-martini.png", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/890-espresso-martini.jpg", real: true },
  { id: "T1_45", code: "T1", name: "Toner Ice White", subtitle: "Use after bleach for perfect blonde results | Lasts up to 16 washes", hex: "#f3e4e3", image: "https://sk-qr.com/assets/images/appearance/LIVE/T1.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/T1.jpg", real: true },
  { id: "109_46", code: "109", name: "Cool Rose", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#eca29a", image: "https://sk-qr.com/assets/images/appearance/LIVE/109-cool-rose.png", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/109-cool-rose.jpg", real: true },
  { id: "B09_47", code: "B09", name: "Spicy Blonde", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#976030", image: "https://sk-qr.com/assets/images/appearance/LIVE/B09.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/B09.jpg", real: true },
  { id: "065_48", code: "065", name: "Spicy Rum", subtitle: "Intense or Pastel look | Up to 15 Washes of Colour Vibrancy", hex: "#4f1806", image: "https://sk-qr.com/assets/images/appearance/LIVE/065-spicy-rum.png", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/065-spicy-rum.jpg", real: true },
  { id: "L76_49", code: "L76", name: "Ultra Violet", subtitle: "2 in 1 Lightening and Colour | Anti-Fading Effect for vibrant results", hex: "#6a0a30", image: "https://sk-qr.com/assets/images/appearance/LIVE/L76.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/L76.jpg", real: true },
  { id: "098_50", code: "098", name: "Steel Silver", subtitle: "Intense or Pastel look | Up to 15 Washes of Colour Vibrancy", hex: "#d7dae6", image: "https://sk-qr.com/assets/images/appearance/LIVE/098.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/098.jpg", real: true },
  { id: "035_51", code: "035", name: "Real Red", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#a00f0a", image: "https://sk-qr.com/assets/images/appearance/LIVE/035__RealRed.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/035__RealRed.jpg", real: true },
  { id: "B14_52", code: "B14", name: "Sparkling Blonde", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#b49169", image: "https://sk-qr.com/assets/images/appearance/LIVE/B14.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/B14.jpg", real: true },
  { id: "010_53", code: "010", name: "Charcoal Brunette", subtitle: "Intense or Pastel look | Up to 15 Washes of Colour Vibrancy", hex: "#180c02", image: "https://sk-qr.com/assets/images/appearance/LIVE/010-charcoal-brunette.png", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/010-charcoal-brunette.jpg", real: true },
  { id: "T2_54", code: "T2", name: "Toner Soft Blonde", subtitle: "Use after bleach for perfect blonde results | Lasts up to 16 washes", hex: "#f0d2a0", image: "https://sk-qr.com/assets/images/appearance/LIVE/T2.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/T2.jpg", real: true },
  { id: "U68_55", code: "U68", name: "Ruby Glaze", subtitle: "Cool tones with Metallic Shine | Anti-Fading Effect", hex: "#580a19", image: "https://sk-qr.com/assets/images/appearance/LIVE/U68.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/U68.jpg", real: true },
  { id: "M06_56", code: "M06", name: "Cocoa Crush", subtitle: "Intense care with oil | Leaves hair healthy & supple", hex: "#44210a", image: "https://sk-qr.com/assets/images/appearance/LIVE/M06.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/M06.jpg", real: true },
  { id: "B16_57", code: "B16", name: "Honey Blonde", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#a08262", image: "https://sk-qr.com/assets/images/appearance/LIVE/B16.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/B16.jpg", real: true },
  { id: "P122_58", code: "P122", name: "Perfect Peach", subtitle: "Creates soft pastel hue | Up to 8 washes", hex: "#f5a77b", image: "https://sk-qr.com/assets/images/appearance/LIVE/P122.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/P122.jpg", real: true },
  { id: "B12_59", code: "B12", name: "Beach Blonde", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#bc8e6f", image: "https://sk-qr.com/assets/images/appearance/LIVE/B12.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/B12.jpg", real: true },
  { id: "U75_60", code: "U75", name: "Midnight Jade", subtitle: "Cool tones with Metallic Shine | Anti-Fading Effect", hex: "#083732", image: "https://sk-qr.com/assets/images/appearance/LIVE/U75.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/U75.jpg", real: true },
  { id: "U72_61", code: "U72", name: "Dusty Silver", subtitle: "Cool tones with Metallic Shine | Anti-Fading Effect", hex: "#69625e", image: "https://sk-qr.com/assets/images/appearance/LIVE/U72.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/U72.jpg", real: true },
  { id: "688_62", code: "6.88", name: "Raspberry Red", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#781018", image: "https://sk-qr.com/assets/images/appearance/LIVE/688.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/688.jpg", real: true },
  { id: "00H_63", code: "00H", name: "Highlight Kit", subtitle: "Anti-Brassiness | Up to 80% less hair breakage", hex: "#978977", image: "https://sk-qr.com/assets/images/appearance/LIVE/00H-highlight-kit.png", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/00H-highlight-kit.jpg", real: true },
  { id: "B10_64", code: "B10", name: "Cool Blonde", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#edc299", image: "https://sk-qr.com/assets/images/appearance/LIVE/B10.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/B10.jpg", real: true },
  { id: "L68_65", code: "L68", name: "Taki Red", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#671112", image: "https://sk-qr.com/assets/images/appearance/LIVE/l68-taki-red.png", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/l68-taki-red.jpg", real: true },
  { id: "099_66", code: "099", name: "Deep Black", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#1b191a", image: "https://sk-qr.com/assets/images/appearance/LIVE/099%20deep%20black.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/099%20deep%20black.jpg", real: true },
  { id: "104_67", code: "104", name: "Cool Lilac", subtitle: "2 in 1 Lightening + Pastel | Long-lasting Pastel Colour", hex: "#d49b9c", image: "https://sk-qr.com/assets/images/appearance/LIVE/104.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/104.jpg", real: true },
  { id: "M01_68", code: "M01", name: "Twilight Black", subtitle: "Intense care with oil | Leaves hair healthy & supple", hex: "#1d2024", image: "https://sk-qr.com/assets/images/appearance/LIVE/M01.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/M01.jpg", real: true },
  { id: "093_69", code: "093", name: "Shocking Pink", subtitle: "Intense or Pastel look | Up to 15 Washes of Colour Vibrancy", hex: "#eb1178", image: "https://sk-qr.com/assets/images/appearance/LIVE/093%20Shocking%20pink.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/093%20Shocking%20pink.jpg", real: true },
  { id: "B15_70", code: "B15", name: "Platinum Blonde", subtitle: "Intense Vibrant Colour | Anti-Fading Effect", hex: "#decdd5", image: "https://sk-qr.com/assets/images/appearance/LIVE/B15.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/B15.jpg", real: true },
  { id: "L77_71", code: "L77", name: "Deep Coral", subtitle: "2 in 1 Lightening and Colour | Anti-Fading Effect for vibrant results", hex: "#c62341", image: "https://sk-qr.com/assets/images/appearance/LIVE/L77.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/L77.jpg", real: true },
  { id: "094_72", code: "094", name: "Purple Punk", subtitle: "Intense or Pastel look | Up to 15 Washes of Colour Vibrancy", hex: "#6c206c", image: "https://sk-qr.com/assets/images/appearance/LIVE/094.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/094.jpg", real: true },
  { id: "093_73", code: "093", name: "Neon Pink", subtitle: "Intense or Pastel look | Up to 12 Washes of Colour Vibrancy", hex: "#f6559b", image: "https://sk-qr.com/assets/images/appearance/LIVE/093%20neon%20pink.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/093%20neon%20pink.jpg", real: true },
  { id: "095_74", code: "095", name: "Ultra Blue", subtitle: "Intense or Pastel look | Up to 12 Washes of Colour Vibrancy", hex: "#1c3a84", image: "https://sk-qr.com/assets/images/appearance/LIVE/095%20Ultra%20blue.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/095%20Ultra%20blue.jpg", real: true },
  { id: "L09_75", code: "L09", name: "Ultra Lightener", subtitle: "Intense care with oil | Leaves hair healthy & supple", hex: "#fbdfb2", image: "https://sk-qr.com/assets/images/appearance/LIVE/L09.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/L09.jpg", real: true },
  { id: "L75_76", code: "L75", name: "Deep Red", subtitle: "2 in 1 Lightening and Colour | Anti-Fading Effect for vibrant results", hex: "#80152a", image: "https://sk-qr.com/assets/images/appearance/LIVE/L75.jpeg", swatchImage: "https://sk-qr.com/assets/images/slider/LIVE/L75.jpg", real: true },
];

/** Воспринимаемая яркость hex-цвета (0..255) — для сортировки палитры. */
function hexLuma(hex: string): number {
  const v = hex.replace('#', '');
  const r = parseInt(v.slice(0, 2), 16) || 0;
  const g = parseInt(v.slice(2, 4), 16) || 0;
  const b = parseInt(v.slice(4, 6), 16) || 0;
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/**
 * Раскладка палитры: ТЁМНЫЕ по краям, СВЕТЛЫЕ (блонд/платина) в ЦЕНТРЕ скролла.
 * Карусель закольцована, поэтому «в конце» = один свайп назад от старта — значит
 * чтобы до светлых нельзя было быстро догортать, их кладём в середину, а тёмные
 * (которые хорошо ложатся) — на внешние позиции (старт и точка заворота).
 *
 * Делаем bitonic-раскладку: сортируем по яркости (тёмные->светлые), затем кладём
 * по очереди на внешние позиции (0, N-1, 1, N-2, ...). Так самые тёмные попадают
 * к краям, а самые светлые сходятся в центр. Сортируем КОПИЮ (без мутации).
 */
const byLuma = [...RAW_PRESETS].sort((a, b) => {
  const dl = hexLuma(a.hex) - hexLuma(b.hex);
  if (Math.abs(dl) > 0.5) return dl; // тёмные -> светлые
  const byCode = a.code.localeCompare(b.code, undefined, { numeric: true, sensitivity: 'base' });
  return byCode !== 0 ? byCode : a.id.localeCompare(b.id);
});
export const PRESETS: ColorPreset[] = (() => {
  const out: ColorPreset[] = new Array(byLuma.length);
  let lo = 0;
  let hi = byLuma.length - 1;
  for (let i = 0; i < byLuma.length; i++) {
    if (i % 2 === 0) out[lo++] = byLuma[i]; // тёмные -> к началу/краю
    else out[hi--] = byLuma[i];             // следующие -> к концу/краю
  }
  return out; // тёмные по краям, светлые в центре
})();

/**
 * LUT-атлас (яркость->цвет по каждому оттенку), построенный из фото прядей.
 * Строка = оттенок (в порядке PRESETS), X = яркость волоса. Шейдер сэмплит его
 * вместо простого HSL-сдвига — это повторяет подход LUT light/dark из референса.
 */
export const LUT = { atlas: '/lut/live-lut.png', rows: PRESETS.length } as const;

/** Индекс пресета в PRESETS (строка LUT). -1 если не найден. */
export function presetRow(id: string): number {
  return PRESETS.findIndex((p) => p.id === id);
}

/** hex (#rrggbb) -> RGB 0..1 для прокидывания в шейдер. Битый hex -> 0. */
export function hexToRgb(hex: string): import('./types').RGB {
  const v = hex.replace('#', '');
  const ch = (s: string) => {
    const n = parseInt(s, 16);
    return Number.isNaN(n) ? 0 : n / 255;
  };
  return { r: ch(v.slice(0, 2)), g: ch(v.slice(2, 4)), b: ch(v.slice(4, 6)) };
}
