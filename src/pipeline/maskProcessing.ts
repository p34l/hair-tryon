/**
 * Постобработка маски волос: адаптивное темпоральное EMA-сглаживание (этап 2),
 * вычитание бороды/лица (этап 1) и подготовка к загрузке в текстуру.
 *
 * EMA здесь motion-compensated: коэффициент сглаживания подстраивается под
 * количество движения в кадре. В покое сглаживаем сильно (стабильный край,
 * не «кипит»), в движении — слабо (нет призрака/запаздывания цвета).
 */

import {
  INFERENCE_SIZE,
  MASK_EMA_ALPHA,
  EMA_ALPHA_STILL,
  EMA_MOTION_GAIN,
} from '../config';

export class MaskProcessor {
  private prev: Float32Array | null = null;
  private out = new Uint8Array(INFERENCE_SIZE * INFERENCE_SIZE);
  // Переиспользуемый буфер под exclusion-результат (никаких new на кадр).
  private excluded = new Uint8Array(INFERENCE_SIZE * INFERENCE_SIZE);

  /** Сбросить накопленное состояние (напр. при перезапуске камеры). */
  reset() {
    this.prev = null;
  }

  /**
   * Этап 2: адаптивное (motion-compensated) экспоненциальное скользящее среднее.
   *
   * Сначала оцениваем движение как среднюю нормированную |current−prev| по всей
   * маске (0..1), затем
   *   alpha = clamp(STILL + motion*GAIN, STILL, MAX)
   * и применяем  mask_t = alpha * current + (1 - alpha) * prev.
   */
  private applyEMA(current: Uint8Array): Uint8Array {
    if (!this.prev || this.prev.length !== current.length) {
      this.prev = Float32Array.from(current);
      // первый кадр — отдать как есть, копировать в out
      for (let i = 0; i < current.length; i++) this.out[i] = current[i];
      return this.out;
    }
    const prev = this.prev;
    const n = current.length;

    // 1) Движение считаем ТОЛЬКО по зоне волос (cur>0 || prev>0): движение фона
    //    не должно раздувать alpha, иначе край волос зря «кипит». Знаменатель —
    //    число активных текселей.
    let diffSum = 0;
    let active = 0;
    for (let i = 0; i < n; i++) {
      const c = current[i];
      const p = prev[i];
      if (c > 0 || p > 0) {
        const d = c - p;
        diffSum += d < 0 ? -d : d;
        active++;
      }
    }
    const motion = active > 0 ? diffSum / (active * 255) : 0;

    // 2) Адаптивный alpha: покой -> STILL, движение -> к MAX.
    let a = EMA_ALPHA_STILL + motion * EMA_MOTION_GAIN;
    if (a < EMA_ALPHA_STILL) a = EMA_ALPHA_STILL;
    if (a > MASK_EMA_ALPHA) a = MASK_EMA_ALPHA;
    const ia = 1 - a;

    // 3) Сглаживание.
    for (let i = 0; i < n; i++) {
      const v = a * current[i] + ia * prev[i];
      prev[i] = v;
      this.out[i] = v;
    }
    return this.out;
  }

  /**
   * Главная точка входа. Порядок: СНАЧАЛА темпоральное EMA по сырой маске волос,
   * ПОТОМ вычитание зоны бороды/лица (жёсткий вырез) — иначе exclusion попадал бы
   * в историю EMA и «тянулся» при движении головы.
   */
  process(
    rawHair: Uint8Array,
    opts: { smooth?: boolean; exclusion?: Uint8Array | null } = {},
  ): Uint8Array {
    let mask = rawHair;

    // Этап 2: адаптивное темпоральное сглаживание (по чистой маске волос).
    if (opts.smooth) {
      mask = this.applyEMA(mask);
    }

    // Этап 1: вычитаем зону бороды/лица — ПОСЛЕ сглаживания.
    if (opts.exclusion) {
      const ex = opts.exclusion;
      if (this.excluded.length !== mask.length) {
        this.excluded = new Uint8Array(mask.length);
      }
      // Не мутируем входной/EMA-буфер, если он переиспользуется как история.
      let target = mask;
      if (target === rawHair) {
        this.excluded.set(rawHair);
        target = this.excluded;
      }
      for (let i = 0; i < target.length; i++) {
        if (ex[i]) target[i] = 0;
      }
      mask = target;
    }

    return mask;
  }
}
