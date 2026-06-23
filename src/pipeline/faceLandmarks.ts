/**
 * Face Landmarker (этап 1): точки лица для построения exclusion-маски нижней
 * части лица, чтобы детерминированно исключить бороду/усы из окраски.
 *
 * Запускается на ГЛАВНОМ потоке (бриф: для PoC так проще, модель лёгкая).
 * В отличие от сегментации, здесь обычный ESM-import работает: проблема
 * "ModuleFactory not set" была специфична для module-воркера (importScripts),
 * а на главном потоке MediaPipe грузится штатно. WASM берём с CDN версии,
 * совпадающей с установленной (config.MODEL.wasmRoot), модель — локально.
 *
 * Идея: по landmarks контура лица (face oval) строим полигон нижней части
 * (от уха до уха по линии челюсти, вверх до уровня носа/усов), растеризуем его
 * в маску. Финальная маска волос = hair * (1 - exclusion) — см. maskProcessing.
 */

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { INFERENCE_SIZE, MODEL } from '../config';

/**
 * Полигон нижней части лица (индексы FaceLandmarker, 478 точек).
 * Нижняя дуга — линия челюсти от правой щеки (454) через подбородок (152)
 * к левой щеке (234). Замыкается по прямой 234→454 на уровне щёк/носа,
 * что накрывает зону бороды, усов, подбородка и нижних щёк.
 */
const LOWER_FACE_POLYGON: number[] = [
  454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
  148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
];

export class FaceMaskBuilder {
  private landmarker: FaceLandmarker | null = null;
  private canvas = new OffscreenCanvas(INFERENCE_SIZE, INFERENCE_SIZE);
  private ctx = this.canvas.getContext('2d', { willReadFrequently: true })!;
  private exclusion = new Uint8Array(INFERENCE_SIZE * INFERENCE_SIZE);
  // Последний timestamp detectForVideo (VIDEO-режим требует возрастающих меток).
  private lastTs = -1;

  get ready() {
    return this.landmarker !== null;
  }

  async init() {
    const fileset = await FilesetResolver.forVisionTasks(MODEL.wasmRoot);
    const modelAssetPath = new URL(MODEL.faceLandmarker, location.origin).href;
    // GPU-делегат намного быстрее (CPU-инференс на главном потоке роняет FPS).
    // Если GPU не поднялся — откат на CPU.
    try {
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
      });
    } catch (e) {
      this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath, delegate: 'CPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
      });
    }
    // Прогрев на холостом кадре: первый detectForVideo компилирует GPU-кернелы.
    // Делаем на экране загрузки, чтобы при старте не было ривка на первом детекте.
    this.warmup();
  }

  /** Прогон по пустому кадру, чтобы скомпилировать кернелы заранее. */
  private warmup() {
    if (!this.landmarker) return;
    try {
      this.ctx.clearRect(0, 0, INFERENCE_SIZE, INFERENCE_SIZE);
      this.lastTs = 0;
      this.landmarker.detectForVideo(this.canvas, this.lastTs);
    } catch {
      /* прогрев необязателен — игнорируем сбой */
    }
  }

  /**
   * Строит маску нижней части лица из landmarks. Возвращает Uint8-буфер
   * (255 в зоне бороды/усов), либо null если лицо не найдено.
   * Синхронный: detectForVideo у tasks-vision возвращает результат сразу.
   */
  buildJawExclusion(
    source: HTMLVideoElement | HTMLImageElement,
    timestamp: number,
  ): Uint8Array | null {
    if (!this.landmarker) return null;

    const S = INFERENCE_SIZE;
    const ctx = this.ctx;

    // Детект на уменьшенном 256-кадре (быстрее; бриф: на уменьшенном кадре).
    // Источник растягивается в квадрат так же, как кадр для сегментации волос,
    // поэтому маски совпадают.
    ctx.drawImage(source, 0, 0, S, S);
    // Строго возрастающий timestamp (после прогрева lastTs уже = 0).
    let ts = timestamp;
    if (ts <= this.lastTs) ts = this.lastTs + 1;
    this.lastTs = ts;
    const result = this.landmarker.detectForVideo(this.canvas, ts);
    const faces = result.faceLandmarks;
    if (!faces || faces.length === 0) return null;

    const lm = faces[0]; // normalized {x,y,z} в [0..1]

    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    LOWER_FACE_POLYGON.forEach((idx, i) => {
      const p = lm[idx];
      const x = p.x * S;
      const y = p.y * S;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill();

    // Читаем нарисованный полигон в Uint8 (берём красный канал как маску).
    const img = ctx.getImageData(0, 0, S, S).data;
    for (let i = 0, p = 0; i < this.exclusion.length; i++, p += 4) {
      this.exclusion[i] = img[p] > 127 ? 255 : 0;
    }
    return this.exclusion;
  }
}
