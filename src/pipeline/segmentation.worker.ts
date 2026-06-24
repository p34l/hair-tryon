/**
 * Web Worker: инференс сегментации волос.
 *
 * ВАЖНО про загрузку MediaPipe здесь:
 * - В module-воркере @mediapipe/tasks-vision падает с "ModuleFactory not set",
 *   потому что его внутренний загрузчик использует importScripts(), которого
 *   в module-воркерах нет (google-ai-edge/mediapipe#5257, #5527).
 * - Поэтому это КЛАССИЧЕСКИЙ воркер (new Worker без type:'module'), а MediaPipe
 *   подгружается через importScripts() из CJS-бандла (локального .js).
 * - В этом файле НЕТ ни одного ESM-import — иначе Vite в dev отдаст воркер как
 *   module и importScripts сломается.
 * - Весь код обёрнут в IIFE: importScripts вливает глобалы минифицированного
 *   бандла в global scope воркера, и без обёртки наши верхнеуровневые const
 *   (напр. короткие имена) конфликтуют с ними ("Identifier 'g' already declared").
 *
 * Почему воркер, а не main thread: segmentForVideo() синхронный и тяжёлый —
 * на главном потоке фризит UI (обязательное условие из брифа).
 */

(function () {
  const INFERENCE_SIZE = 256;
  const HAIR_CLASS = 1; // SelfieMulticlass: 0 bg,1 hair,2 body-skin,3 face-skin,4 clothes,5 other
  // Бандл и WASM лежат локально в public/mediapipe (скопированы из node_modules).
  // CDN отдаёт .cjs с MIME application/node, который importScripts отвергает;
  // локальный .js Vite отдаёт как application/javascript. Версия = node_modules.
  const CJS_URL = '/mediapipe/vision_bundle.js';
  const WASM_ROOT = '/mediapipe/wasm';

  // Шим CommonJS-окружения, чтобы importScripts CJS-бандла записал экспорты.
  const glob = self as any;
  glob.module = { exports: {} };
  glob.exports = glob.module.exports;
  importScripts(CJS_URL);
  const vision = glob.module.exports;
  const ImageSegmenter = vision.ImageSegmenter;
  const FilesetResolver = vision.FilesetResolver;

  let segmenter: any = null;

  const offscreen = new OffscreenCanvas(INFERENCE_SIZE, INFERENCE_SIZE);
  const offCtx = offscreen.getContext('2d', { willReadFrequently: true })!;

  let maskBuffer = new Uint8Array(INFERENCE_SIZE * INFERENCE_SIZE);
  // Последний использованный timestamp segmentForVideo (VIDEO-режим требует
  // строго возрастающих меток). Прогрев занимает 0, реальные кадры идут дальше.
  let lastTs = -1;

  function post(msg: any, transfer?: Transferable[]) {
    (self as any).postMessage(msg, transfer ?? []);
  }

  async function init(modelPath: string) {
    try {
      const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
      segmenter = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelPath, delegate: 'GPU' },
        runningMode: 'VIDEO',
        // Мягкая (confidence) маска класса hair — без бинарного порога (край не
        // «кипит»). categoryMask НЕ запрашиваем: её argmax — лишняя работа каждый
        // инференс (грелся GPU), а используем мы только confidenceMasks[1].
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
      // Прогрев на холостом кадре: ПЕРВЫЙ инференс компилирует GPU-кернелы —
      // делаем это сейчас, на экране загрузки, чтобы при старте камеры не было
      // лага. 'ready' шлём только после прогрева (с подстраховкой по таймауту).
      warmupThenReady();
    } catch (err) {
      post({ type: 'error', message: String(err) });
    }
  }

  function warmupThenReady() {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      post({ type: 'ready' });
    };
    try {
      offCtx.clearRect(0, 0, INFERENCE_SIZE, INFERENCE_SIZE);
      lastTs = 0;
      segmenter.segmentForVideo(offscreen, lastTs, (result: any) => {
        result.close?.();
        finish();
      });
      // если колбэк не пришёл (маловероятно) — всё равно отдаём готовность
      setTimeout(finish, 2500);
    } catch {
      finish();
    }
  }

  function segment(pixels: ArrayBuffer, w: number, h: number, timestamp: number) {
    if (!segmenter) return;
    // Кадр приходит как СЫРЫЕ ПИКСЕЛИ (ImageData buffer), а не ImageBitmap:
    // createImageBitmap течёт по памяти на iOS Safari (закрытые битмапы не
    // освобождаются) — за ~30с упирается в лимит вкладки и FPS падает без отката.
    // putImageData в reused-canvas аллокаций не плодит.
    const id = new ImageData(new Uint8ClampedArray(pixels), w, h);
    offCtx.putImageData(id, 0, 0);

    // Гарантируем строго возрастающий timestamp (после прогрева и в принципе).
    let ts = timestamp;
    if (ts <= lastTs) ts = lastTs + 1;
    lastTs = ts;

    try {
      segmenter.segmentForVideo(offscreen, ts, (result: any) => {
        // Предпочитаем мягкую confidence-маску класса hair: вероятность 0..1
        // без порога. Это убирает бинарное «мерцание» на границе прядей.
        const confMasks = result.confidenceMasks;
        const hairConf = confMasks && confMasks[HAIR_CLASS];
        let wrote = false;

        if (hairConf) {
          const probs: Float32Array = hairConf.getAsFloat32Array();
          if (maskBuffer.length !== probs.length) maskBuffer = new Uint8Array(probs.length);
          for (let i = 0; i < probs.length; i++) {
            // Контраст вероятности перед квантованием: smoothstep(0.2,0.8) гасит
            // низкоуверенный спекл (фон) и подтягивает уверенное ядро, сохраняя
            // мягкий градиент на переходе 0.2..0.8 (согласовано с MASK_EDGE_LOW/HIGH
            // в шейдере). Без бинаризации — край не «кипит».
            let t = (probs[i] - 0.2) / 0.6;
            t = t < 0 ? 0 : t > 1 ? 1 : t;
            const p = t * t * (3.0 - 2.0 * t);
            const v = p * 255.0;
            maskBuffer[i] = v < 0 ? 0 : v > 255 ? 255 : v;
          }
          wrote = true;
        } else if (result.categoryMask) {
          // Фолбэк: старая бинарная логика из категориальной маски.
          const src: Uint8Array = result.categoryMask.getAsUint8Array();
          if (maskBuffer.length !== src.length) maskBuffer = new Uint8Array(src.length);
          for (let i = 0; i < src.length; i++) {
            maskBuffer[i] = src[i] === HAIR_CLASS ? 255 : 0;
          }
          wrote = true;
        }

        // Освобождаем нативные буферы результата (масок + сам результат).
        result.close?.();

        if (!wrote) return;
        const out = maskBuffer.slice();
        post({ type: 'mask', data: out, width: INFERENCE_SIZE, height: INFERENCE_SIZE }, [out.buffer]);
      });
    } catch (err) {
      post({ type: 'error', message: 'segmentForVideo: ' + String(err) });
    }
  }

  self.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    if (msg.type === 'init') init(msg.modelPath);
    else if (msg.type === 'segment') segment(msg.pixels, msg.width, msg.height, msg.timestamp);
  };
})();
