/**
 * Сегментация волос НА ГЛАВНОМ ПОТОКЕ в общем GL-контексте с рендером.
 *
 * Почему снова на главном потоке (раньше был воркер): тяжёлым был не инференс
 * (~7мс), а синхронный GPU->CPU readback маски (getAsFloat32Array, ~75мс). В новой
 * архитектуре readback'а НЕТ: маска остаётся текстурой на GPU (getAsWebGLTexture)
 * и сразу идёт в recolor-шейдер. Поэтому segmentForVideo на главном потоке больше
 * не фризит UI, а общий с рендером контекст — обязателен, иначе текстуру маски
 * нельзя использовать в нашем шейдере (текстуры не пересекают GL-контексты).
 *
 * Сегментатор создаётся с опцией `canvas` = canvas рендера: MediaPipe инициализирует
 * (или переиспользует) WebGL2-контекст ЭТОГО canvas, и getAsWebGLTexture() отдаёт
 * текстуру, валидную в нём же.
 */

import { ImageSegmenter, FilesetResolver } from '@mediapipe/tasks-vision';
import { MODEL } from '../config';

export const HAIR_CLASS = 1; // и SelfieMulticlass, и hair_segmenter держат hair на 1

type Delegate = 'GPU' | 'CPU';
interface Rung { model: string; delegate: Delegate }

/**
 * Колбэк получает маску одним из двух способов (см. usePixels):
 *  • tex  — GPU-текстура (Android, zero-readback);
 *  • data — Uint8Array 0..255 (iOS: getAsWebGLTexture там течёт памятью, поэтому
 *           читаем пиксели и заливаем в свою постоянную текстуру — без утечки).
 * null — маски нет (сбой/смена ступени).
 */
export type MaskPayload =
  | { tex: WebGLTexture; data?: undefined; width: number; height: number }
  | { tex?: undefined; data: Uint8Array; width: number; height: number };
export type MaskCallback = (payload: MaskPayload | null) => void;

export class HairSegmenter {
  private canvas: HTMLCanvasElement | OffscreenCanvas;
  private segmenter: ImageSegmenter | null = null;
  private fileset: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;
  // iOS: отдавать маску пикселями (getAsUint8Array), а НЕ getAsWebGLTexture —
  // последний на iOS течёт GPU-памятью (новая текстура каждый кадр).
  private usePixels: boolean;

  // Лестница фолбэка делегата (тот же краш-фикс, что в воркерной версии):
  //  1) мультиклас @ GPU  — лучшая якість маски (десктоп/iOS, исправные Android);
  //  2) hair-модель @ GPU — её GPU-путь устойчив там, где мультиклас на GPU падает
  //     ("confidence_mask_count 0 vs 6"); быстро, без CPU;
  //  3) hair-модель @ CPU — крайний случай, лишь бы не падать.
  private ladder: Rung[] = [];
  private rung = 0;
  private lastTs = -1;
  private _backend = '';

  constructor(canvas: HTMLCanvasElement | OffscreenCanvas, usePixels = false) {
    this.canvas = canvas;
    this.usePixels = usePixels;
  }

  get backend(): string { return this._backend; }

  private backendLabel(): string {
    const c = this.ladder[this.rung];
    if (!c) return '';
    const m = c.model.indexOf('hair') >= 0 ? 'hair' : 'multi';
    return m + '@' + c.delegate;
  }

  private async create(): Promise<void> {
    if (!this.fileset) {
      this.fileset = await FilesetResolver.forVisionTasks(MODEL.wasmRoot);
    }
    if (this.segmenter) {
      try { this.segmenter.close(); } catch { /* ignore */ }
      this.segmenter = null;
    }
    const cfg = this.ladder[this.rung];
    this.segmenter = await ImageSegmenter.createFromOptions(this.fileset, {
      baseOptions: { modelAssetPath: cfg.model, delegate: cfg.delegate },
      // Общий GL-контекст с рендером — ключ к zero-readback (getAsWebGLTexture).
      canvas: this.canvas,
      runningMode: 'VIDEO',
      outputCategoryMask: false,
      outputConfidenceMasks: true,
    });
    this._backend = this.backendLabel();
  }

  /** Спуститься на следующую ступень после сбоя текущей. true — если переключились. */
  private async fallbackNext(): Promise<boolean> {
    if (this.rung >= this.ladder.length - 1) return false;
    this.rung++;
    this.lastTs = -1;
    try {
      await this.create();
      return true;
    } catch {
      return this.fallbackNext();
    }
  }

  /** Инициализация. modelPath/hairModelPath — абсолютные URL моделей. */
  async init(modelPath: string, hairModelPath: string): Promise<void> {
    this.ladder = [
      { model: modelPath, delegate: 'GPU' },
      { model: hairModelPath, delegate: 'GPU' },
      { model: hairModelPath, delegate: 'CPU' },
    ];
    this.rung = 0;
    try {
      await this.create();
    } catch (err) {
      if (!(await this.fallbackNext())) throw err;
    }
  }

  /**
   * Сегментирует кадр и отдаёт текстуру маски в колбэке. Колбэк вызывается
   * СИНХРОННО внутри segmentForVideo — текстура валидна только внутри него,
   * поэтому рендерить надо там же (см. CameraView). При сбое делегата —
   * спускаемся по лестнице и пропускаем этот кадр (следующий пойдёт на новом).
   */
  segment(source: TexImageSource, timestamp: number, cb: MaskCallback): void {
    const seg = this.segmenter;
    if (!seg) { cb(null); return; }

    let ts = timestamp;
    if (ts <= this.lastTs) ts = this.lastTs + 1;
    this.lastTs = ts;

    try {
      seg.segmentForVideo(source as unknown as HTMLVideoElement, ts, (result) => {
        // try/finally КРИТИЧНО: если cb()->render() кинет в общем GL-контексте,
        // без finally result.close() не дойдёт и маска (GPU-текстура) утечёт КАЖДЫЙ
        // такой кадр → FPS падает и не встаёт. close() гарантируем всегда.
        try {
          const masks = result.confidenceMasks;
          const hair = masks && masks[HAIR_CLASS];
          if (hair) {
            if (this.usePixels) {
              // iOS: пиксели (64КБ readback) — БЕЗ getAsWebGLTexture (он там течёт).
              cb({ data: hair.getAsUint8Array(), width: hair.width, height: hair.height });
            } else {
              // Android: текстура остаётся на GPU — нуль readback.
              cb({ tex: hair.getAsWebGLTexture(), width: hair.width, height: hair.height });
            }
          } else {
            cb(null);
          }
        } finally {
          result.close();
        }
      });
    } catch {
      // Сбой делегата на реальном кадре — на следующую ступень; этот кадр пропускаем.
      cb(null);
      void this.fallbackNext();
    }
  }

  /** Тёплый прогон на пустом кадре: компилирует GPU-кернелы до старта камеры. */
  warmup(): void {
    const seg = this.segmenter;
    if (!seg) return;
    try {
      const c = new OffscreenCanvas(64, 64);
      c.getContext('2d');
      this.lastTs = 0;
      seg.segmentForVideo(c as unknown as HTMLVideoElement, 0, (r) => r.close());
    } catch {
      void this.fallbackNext();
    }
  }

  close(): void {
    try { this.segmenter?.close(); } catch { /* ignore */ }
    this.segmenter = null;
  }
}
