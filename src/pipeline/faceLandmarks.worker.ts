/**
 * Web Worker: FaceLandmarker для exclusion-маски бороды/нижней части лица.
 *
 * Почему воркер (а не главный поток, как было раньше):
 * - detectForVideo() — СИНХРОННЫЙ инференс. На главном потоке он блокировал
 *   рендер. Особенно на iOS Safari, где НЕТ requestIdleCallback → код падал на
 *   setTimeout и гонял инференс прямо в основном потоке, роняя FPS. На Mac (где
 *   requestIdleCallback есть) этого не было — отсюда «на маке норм, на айфоне нет».
 * - Здесь инференс и растеризация полигона (getImageData) полностью вне main —
 *   рендер больше никогда не ждёт лэндмарки.
 *
 * Загрузка MediaPipe — как в segmentation.worker: КЛАССИЧЕСКИЙ воркер,
 * importScripts CJS-бандла, весь код в IIFE, без ESM-import.
 */

(function () {
  const INFERENCE_SIZE = 256;
  const CJS_URL = '/mediapipe/vision_bundle.js';
  const WASM_ROOT = '/mediapipe/wasm';

  const glob = self as any;
  glob.module = { exports: {} };
  glob.exports = glob.module.exports;
  importScripts(CJS_URL);
  const vision = glob.module.exports;
  const FaceLandmarker = vision.FaceLandmarker;
  const FilesetResolver = vision.FilesetResolver;

  // Нижняя часть лица (индексы FaceLandmarker, 478 точек): линия челюсти от щеки
  // через подбородок к щеке — накрывает бороду/усы/подбородок.
  const LOWER_FACE_POLYGON = [
    454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
  ];

  let landmarker: any = null;
  let lastTs = -1;
  const offscreen = new OffscreenCanvas(INFERENCE_SIZE, INFERENCE_SIZE);
  const offCtx = offscreen.getContext('2d', { willReadFrequently: true })!;
  let exclusion = new Uint8Array(INFERENCE_SIZE * INFERENCE_SIZE);

  function post(msg: any, transfer?: Transferable[]) {
    (self as any).postMessage(msg, transfer ?? []);
  }

  async function init(modelPath: string) {
    try {
      const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
      try {
        landmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: modelPath, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numFaces: 1,
        });
      } catch {
        landmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: modelPath, delegate: 'CPU' },
          runningMode: 'VIDEO',
          numFaces: 1,
        });
      }
      // Прогрев: первый detectForVideo компилирует кернелы (делаем на загрузке).
      try {
        offCtx.clearRect(0, 0, INFERENCE_SIZE, INFERENCE_SIZE);
        lastTs = 0;
        landmarker.detectForVideo(offscreen, lastTs);
      } catch {
        /* прогрев необязателен */
      }
      post({ type: 'face-ready' });
    } catch (err) {
      post({ type: 'face-error', message: String(err) });
    }
  }

  function detect(pixels: ArrayBuffer, w: number, h: number, timestamp: number) {
    if (!landmarker) return;
    const S = INFERENCE_SIZE;
    // Сырые пиксели вместо ImageBitmap (см. segmentation.worker — течёт на iOS).
    const id = new ImageData(new Uint8ClampedArray(pixels), w, h);
    offCtx.putImageData(id, 0, 0);

    // Строго возрастающий timestamp (VIDEO-режим).
    let ts = timestamp;
    if (ts <= lastTs) ts = lastTs + 1;
    lastTs = ts;

    let result: any;
    try {
      result = landmarker.detectForVideo(offscreen, ts);
    } catch (err) {
      post({ type: 'face-error', message: 'detectForVideo: ' + String(err) });
      return;
    }

    const faces = result && result.faceLandmarks;
    if (!faces || faces.length === 0) {
      post({ type: 'face-mask', data: null });
      return;
    }

    // Растеризуем полигон нижней части лица в маску (всё в воркере, вне main).
    const lm = faces[0];
    offCtx.clearRect(0, 0, S, S);
    offCtx.fillStyle = '#fff';
    offCtx.beginPath();
    for (let i = 0; i < LOWER_FACE_POLYGON.length; i++) {
      const p = lm[LOWER_FACE_POLYGON[i]];
      const x = p.x * S;
      const y = p.y * S;
      if (i === 0) offCtx.moveTo(x, y);
      else offCtx.lineTo(x, y);
    }
    offCtx.closePath();
    offCtx.fill();

    const img = offCtx.getImageData(0, 0, S, S).data;
    if (exclusion.length !== S * S) exclusion = new Uint8Array(S * S);
    for (let i = 0, p = 0; i < exclusion.length; i++, p += 4) {
      exclusion[i] = img[p] > 127 ? 255 : 0;
    }

    const out = exclusion.slice();
    post({ type: 'face-mask', data: out, width: S, height: S }, [out.buffer]);
  }

  self.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    if (msg.type === 'init') init(msg.modelPath);
    else if (msg.type === 'detect') detect(msg.pixels, msg.width, msg.height, msg.timestamp);
  };
})();
