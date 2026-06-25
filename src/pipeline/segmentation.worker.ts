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

// ДИАГНОСТИКА: ORT логирует проблемы WebGPU в console — перехватываем, чтобы
// показать причину «пустой маски» на экране телефона (без DevTools).
ort.env.logLevel = 'warning';
const ortLogs: string[] = [];
const _warn = console.warn.bind(console);
const _err = console.error.bind(console);
console.warn = (...a: any[]) => { ortLogs.push('W ' + a.map(String).join(' ').slice(0, 160)); _warn(...a); };
console.error = (...a: any[]) => { ortLogs.push('E ' + a.map(String).join(' ').slice(0, 160)); _err(...a); };

function post(msg: any, transfer?: Transferable[]) {
  (self as any).postMessage(msg, transfer ?? []);
}

async function init(modelUrl: string) {
  const diag: string[] = [];
  try {
    // Пробуем WebGPU и ПОДРОБНО смотрим, что он отдаёт на тестовом кадре.
    const s = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
    });
    inputName = s.inputNames[0]; outputName = s.outputNames[0];
    diag.push('wgpu session ok; out=' + outputName);
    const warm = new ort.Tensor('float32', new Float32Array(SIZE * SIZE * 3), [1, SIZE, SIZE, 3]);
    const o = await s.run({ [inputName]: warm });
    const ot = o[outputName] as any;
    diag.push('loc=' + (ot.location || '?') + ' dlen=' + (ot.data ? ot.data.length : 'null'));
    let d: Float32Array | null = ot.data;
    if (ot.location && ot.location !== 'cpu') {
      try { d = (await ot.getData(true)) as Float32Array; diag.push('getData len=' + (d ? d.length : 'null')); }
      catch (e) { diag.push('getData ERR ' + String(e).slice(0, 120)); d = null; }
    }
    if (d && d.length) {
      let mn = d[0], mx = d[0];
      for (let i = 1; i < d.length; i++) { const v = d[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
      diag.push('min=' + mn.toFixed(3) + ' max=' + mx.toFixed(3));
    }
    session = s; backend = 'onnx@webgpu';
  } catch (e) {
    diag.push('wgpu ERR ' + String(e).slice(0, 200));
    try {
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'], graphOptimizationLevel: 'all',
      });
      inputName = session.inputNames[0]; outputName = session.outputNames[0];
      backend = 'onnx@wasm';
    } catch (e2) {
      post({ type: 'error', message: 'ORT init: ' + String(e2) });
      return;
    }
  }
  // Диагностику + последние логи ORT — на экран (msg.type='diag').
  post({ type: 'diag', message: diag.join(' | ') + '  ||LOGS|| ' + ortLogs.slice(-8).join(' ;; ') });
  post({ type: 'ready', backend });
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
