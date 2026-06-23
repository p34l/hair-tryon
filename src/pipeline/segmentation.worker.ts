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

  function post(msg: any, transfer?: Transferable[]) {
    (self as any).postMessage(msg, transfer ?? []);
  }

  async function init(modelPath: string) {
    try {
      const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
      segmenter = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelPath, delegate: 'GPU' },
        runningMode: 'VIDEO',
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
      post({ type: 'ready' });
    } catch (err) {
      post({ type: 'error', message: String(err) });
    }
  }

  function segment(bitmap: ImageBitmap, timestamp: number) {
    if (!segmenter) {
      bitmap.close();
      return;
    }
    offCtx.drawImage(bitmap, 0, 0, INFERENCE_SIZE, INFERENCE_SIZE);
    bitmap.close();

    try {
      segmenter.segmentForVideo(offscreen, timestamp, (result: any) => {
        const categoryMask = result.categoryMask;
        if (!categoryMask) return;
        const src: Uint8Array = categoryMask.getAsUint8Array();

        if (maskBuffer.length !== src.length) maskBuffer = new Uint8Array(src.length);
        for (let i = 0; i < src.length; i++) {
          maskBuffer[i] = src[i] === HAIR_CLASS ? 255 : 0;
        }
        categoryMask.close();

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
    else if (msg.type === 'segment') segment(msg.bitmap, msg.timestamp);
  };
})();
