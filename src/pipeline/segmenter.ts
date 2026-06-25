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

/** Колбэк получает текстуру маски (на GPU) + её размеры, либо null если маски нет. */
export type MaskCallback = (tex: WebGLTexture | null, width: number, height: number) => void;

export class HairSegmenter {
  private canvas: HTMLCanvasElement | OffscreenCanvas;
  private segmenter: ImageSegmenter | null = null;
  private fileset: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;

  // Лестница фолбэка делегата (тот же краш-фикс, что в воркерной версии):
  //  1) мультиклас @ GPU  — лучшая якість маски (десктоп/iOS, исправные Android);
  //  2) hair-модель @ GPU — её GPU-путь устойчив там, где мультиклас на GPU падает
  //     ("confidence_mask_count 0 vs 6"); быстро, без CPU;
  //  3) hair-модель @ CPU — крайний случай, лишь бы не падать.
  private ladder: Rung[] = [];
  private rung = 0;
  private lastTs = -1;
  private _backend = '';

  constructor(canvas: HTMLCanvasElement | OffscreenCanvas) {
    this.canvas = canvas;
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
    if (!seg) { cb(null, 0, 0); return; }

    let ts = timestamp;
    if (ts <= this.lastTs) ts = this.lastTs + 1;
    this.lastTs = ts;

    try {
      seg.segmentForVideo(source as unknown as HTMLVideoElement, ts, (result) => {
        const masks = result.confidenceMasks;
        const hair = masks && masks[HAIR_CLASS];
        if (hair) {
          // Текстура остаётся на GPU — никакого getAs*Array (нуль readback).
          const tex = hair.getAsWebGLTexture();
          cb(tex, hair.width, hair.height);
        } else {
          cb(null, 0, 0);
        }
        result.close();
      });
    } catch {
      // Сбой делегата на реальном кадре — на следующую ступень; этот кадр пропускаем.
      cb(null, 0, 0);
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
