/**
 * Постобработка маски волос: EMA-сглаживание между кадрами (этап 2),
 * вычитание бороды/лица (этап 1) и подготовка к загрузке в текстуру.
 *
 * На этапе 0 используется только passthrough — сырая маска идёт в шейдер
 * как есть. Логика EMA/exclusion включается на следующих этапах.
 */

import { INFERENCE_SIZE, MASK_EMA_ALPHA } from '../config';

export class MaskProcessor {
  private prev: Float32Array | null = null;
  private out = new Uint8Array(INFERENCE_SIZE * INFERENCE_SIZE);

  /** Сбросить накопленное состояние (напр. при перезапуске камеры). */
  reset() {
    this.prev = null;
  }

  /**
   * Этап 2: экспоненциальное скользящее среднее.
   * mask_t = alpha * current + (1 - alpha) * prev
   */
  private applyEMA(current: Uint8Array): Uint8Array {
    const a = MASK_EMA_ALPHA;
    if (!this.prev || this.prev.length !== current.length) {
      this.prev = Float32Array.from(current);
    }
    const prev = this.prev;
    for (let i = 0; i < current.length; i++) {
      const v = a * current[i] + (1 - a) * prev[i];
      prev[i] = v;
      this.out[i] = v;
    }
    return this.out;
  }

  /**
   * Главная точка входа. На этапе 0 — просто возвращает сырую маску.
   * На этапе 1 сюда добавится вычитание exclusion-маски (борода/лицо),
   * на этапе 2 включится EMA. Флаги ниже включаются по мере готовности этапов.
   */
  process(
    rawHair: Uint8Array,
    opts: { smooth?: boolean; exclusion?: Uint8Array | null } = {},
  ): Uint8Array {
    let mask = rawHair;

    // Этап 1: вычитаем зону бороды/лица — там маска волос обнуляется.
    if (opts.exclusion) {
      const ex = opts.exclusion;
      const result = new Uint8Array(mask.length);
      for (let i = 0; i < mask.length; i++) {
        result[i] = ex[i] ? 0 : mask[i];
      }
      mask = result;
    }

    // Этап 2: темпоральное сглаживание.
    if (opts.smooth) {
      mask = this.applyEMA(mask);
    }

    return mask;
  }
}
