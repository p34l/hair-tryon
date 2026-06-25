/**
 * Web Worker: сегментация волос через ONNX Runtime Web (ВМЕСТО MediaPipe).
 *
 * Почему ORT, а не MediaPipe: рантайм MediaPipe на iOS Safari течёт GPU/WASM-памятью
 * при покадровом инференсе (доказано: не зависит от архитектуры/версии — inf растёт
 * со временем, лечится только reload). ORT — другой рантайм, со своим управлением
 * памятью; мы выкидываем протекающий MediaPipe, сохраняя весь recolor/UI/маску 256².
 *
 * Это МОДУЛЬНЫЙ воркер (type:'module') — ORT грузится обычным ESM-import. WASM-бинарь
 * ORT берём с CDN (ort.env.wasm.wasmPaths), чтобы не возиться с бандлингом .wasm.
 *
 * Модель: selfie_multiclass_256x256, сконвертированная tflite -> onnx (tf2onnx).
 *  вход  input  [1,256,256,3] float, RGB, нормализация [0,1];
 *  выход Identity [1,256,256,6] float ЛОГИТЫ 6 классов (NHWC). hair = класс 1.
 * Маску волос получаем softmax по 6 классам и берём канал hair.
 */

import * as ort from 'onnxruntime-web';

const SIZE = 256;
const HAIR = 1;        // SelfieMulticlass: 0 bg,1 hair,2 body,3 face,4 clothes,5 other
const CLASSES = 6;

// WASM-бинарь ORT берём со СВОЕГО origin (/ort/). Файлы в public/ort.
ort.env.wasm.wasmPaths = '/ort/';
// ОДИН поток: на мобильных многопоточность дала НЕ ускорение, а замедление
// (inf 95->300мс — накладные на спавн/синхронизацию потоков > выигрыша) и затор.
ort.env.wasm.numThreads = 1;

let session: ort.InferenceSession | null = null;
let inputName = 'input_29';
let outputName = 'Identity';
let inputBuf = new Float32Array(SIZE * SIZE * 3);
let maskBuf = new Uint8Array(SIZE * SIZE);
let busy = false;
let backend = 'onnx';

function post(msg: any, transfer?: Transferable[]) {
  (self as any).postMessage(msg, transfer ?? []);
}

async function init(modelUrl: string) {
  // WebGPU EP (инференс на GPU, ~100мс и быстрее): на мобильном WASM ~300мс — слишком
  // медленно. Раньше WebGPU ломал маску из-за resize-оп `tf_half_pixel_for_nn`; модель
  // ПРОПАТЧЕНА (resize -> linear/half_pixel), маска идентична. Фолбэк на WASM где нет WebGPU.
  // WebGPU EP (GPU, ~100мс; WASM на моб. ~300мс). Гипотеза по «пустой маске»: выход
  // оставался в GPU-буфере и .data был пуст — теперь явно скачиваем через getData()
  // в segment(). Фолбэк на WASM где WebGPU нет.
  const tryEP = (ep: 'webgpu' | 'wasm') =>
    ort.InferenceSession.create(modelUrl, { executionProviders: [ep], graphOptimizationLevel: 'all' });
  try {
    try {
      session = await tryEP('webgpu');
      backend = 'onnx@webgpu';
    } catch {
      session = await tryEP('wasm');
      backend = 'onnx@wasm';
    }
    inputName = session.inputNames[0] ?? inputName;
    outputName = session.outputNames[0] ?? outputName;
    // Прогрев: первый run компилирует кернелы/аллоцирует — на пустом кадре.
    try {
      const warm = new ort.Tensor('float32', new Float32Array(SIZE * SIZE * 3), [1, SIZE, SIZE, 3]);
      await session.run({ [inputName]: warm });
    } catch { /* прогрев необязателен */ }
    post({ type: 'ready', backend });
  } catch (err) {
    post({ type: 'error', message: 'ORT init: ' + String(err) });
  }
}

async function segment(pixels: ArrayBuffer, w: number, h: number) {
  if (!session || busy) return;
  busy = true;
  const t0 = performance.now();
  try {
    // RGBA пиксели -> NHWC float [0,1], только RGB. (w,h обычно 256.)
    const px = new Uint8ClampedArray(pixels);
    const N = w * h;
    if (inputBuf.length !== N * 3) inputBuf = new Float32Array(N * 3);
    for (let i = 0, p = 0; i < N; i++, p += 4) {
      inputBuf[i * 3] = px[p] / 255;
      inputBuf[i * 3 + 1] = px[p + 1] / 255;
      inputBuf[i * 3 + 2] = px[p + 2] / 255;
    }
    const tensor = new ort.Tensor('float32', inputBuf, [1, h, w, 3]);
    const out = await session.run({ [inputName]: tensor });
    // На WebGPU выход может лежать в GPU-буфере — .data пуст. Тогда явно скачиваем
    // через getData(). На WASM location='cpu' и .data готов.
    const ot = out[outputName] as any;
    let logits: Float32Array;
    if (ot.location && ot.location !== 'cpu') {
      logits = (await ot.getData(true)) as Float32Array;
    } else {
      logits = ot.data as Float32Array;
    }

    if (maskBuf.length !== N) maskBuf = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      // softmax по 6 классам пикселя -> вероятность hair.
      const base = i * CLASSES;
      let mx = logits[base];
      for (let c = 1; c < CLASSES; c++) { const v = logits[base + c]; if (v > mx) mx = v; }
      let sum = 0;
      let hairExp = 0;
      for (let c = 0; c < CLASSES; c++) {
        const e = Math.exp(logits[base + c] - mx);
        sum += e;
        if (c === HAIR) hairExp = e;
      }
      const prob = hairExp / sum; // 0..1
      const v = prob * 255;
      maskBuf[i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }

    const data = maskBuf.slice();
    post(
      { type: 'mask', data, width: w, height: h, infMs: performance.now() - t0, backend },
      [data.buffer],
    );
  } catch (err) {
    post({ type: 'error', message: 'ORT segment: ' + String(err) });
  } finally {
    busy = false;
  }
}

self.onmessage = (e: MessageEvent) => {
  const msg = e.data;
  if (msg.type === 'init') init(msg.modelPath);
  else if (msg.type === 'segment') segment(msg.pixels, msg.width, msg.height);
};
