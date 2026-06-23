/** Общие типы проекта. */

/** RGB в диапазоне 0..1 — удобно прокидывать прямо в шейдер как uniform. */
export interface RGB {
  r: number;
  g: number;
  b: number;
}

/** Режим интенсивности. Управляет strength и насыщенностью в шейдере. */
export type Intensity = 'intense' | 'pastel';

/**
 * Один цвет-пресет = одна «краска» из референса.
 * swatch — цвет кружочка в UI (может отличаться от targetColor превью-стиля),
 * box — необязательная картинка коробки продукта.
 */
export interface ColorPreset {
  id: string;
  /** Код оттенка, напр. "092". */
  code: string;
  /** Название оттенка, напр. "Pillar Box Red". */
  name: string;
  /** Подпись под названием, напр. "Intense or Pastel look". */
  subtitle: string;
  /** Целевой цвет волос в hex (#rrggbb) — вариант Intense (Vibrant). */
  hex: string;
  /** Цвет для режима Pastel (вариант Pastel из AR-API). Если нет — берётся hex. */
  pastelHex?: string;
  /** Цвет кружочка-сватча в UI (по умолчанию = hex). */
  swatch?: string;
  /** Реальное фото пряди для кружочка-сватча (если есть — показываем его). */
  swatchImage?: string;
  /** true — цвет получен из официального Schwarzkopf AR-API; false — курирован. */
  real?: boolean;
  /**
   * URL реального фото коробки продукта. Если не задан или не загрузился —
   * показываем сгенерированный SVG-мокап коробки (см. ColorPicker).
   */
  image?: string;
}

/** Сообщение из основного потока в worker: кадр на сегментацию. */
export interface SegmentRequest {
  type: 'segment';
  bitmap: ImageBitmap;
  /** Метка времени видеокадра для segmentForVideo. */
  timestamp: number;
}

/** Сообщение «инициализируйся» в worker. */
export interface InitRequest {
  type: 'init';
  modelPath: string;
  wasmRoot: string;
}

export type WorkerRequest = SegmentRequest | InitRequest;

/** Worker → main: маска готова. data — канал hair, INFERENCE_SIZE². */
export interface MaskResult {
  type: 'mask';
  data: Uint8Array;
  width: number;
  height: number;
}

/** Worker → main: статус (готов / ошибка). */
export interface StatusResult {
  type: 'ready' | 'error';
  message?: string;
}

export type WorkerResponse = MaskResult | StatusResult;
