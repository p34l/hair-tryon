/**
 * Web Worker: сегментация волос через ONNX Runtime Web на модели MediaPipe
 * SelfieMulticlass (tflite -> onnx, конвертирована в NCHW + патч resize + softmax/
 * выбор класса hair в графе). Это ТА ЖЕ модель, что давала лучшую маску в MediaPipe-
 * пайплайне — но БЕЗ протекающего MediaPipe-рантайма (ORT не течёт на iOS).
 *
 * Модель: вход [1,3,256,256] NCHW, нормализация [0,1]; в граф добавлен softmax по
 * каналам + выбор класса hair(1) -> выход [1,256,256,1] вероятность волос (256²,
 * сразу под recolor-шейдер). У неё есть attention-транспозы → WebGPU ~120мс (но в
 * воркере это не дропает FPS), зато качество маски как в пайплайне.
 */

import * as ort from 'onnxruntime-web';

const SIZE = 256;

ort.env.wasm.wasmPaths = '/ort/';
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'warning';
const ortLogs: string[] = [];
const _warn = console.warn.bind(console);
const _err = console.error.bind(console);
console.warn = (...a: any[]) => { ortLogs.push('W ' + a.map(String).join(' ').slice(0, 140)); _warn(...a); };
console.error = (...a: any[]) => { ortLogs.push('E ' + a.map(String).join(' ').slice(0, 140)); _err(...a); };

let session: ort.InferenceSession | null = null;
let inputName = 'input_29';
let outputName = 'hair';
let backend = 'onnx';
let busy = false;

const inputBuf = new Float32Array(3 * SIZE * SIZE);
const maskBuf = new Uint8Array(SIZE * SIZE);

function post(msg: any, transfer?: Transferable[]) {
  (self as any).postMessage(msg, transfer ?? []);
}

async function init(modelUrl: string) {
  const diag: string[] = [];
  try {
    const s = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
    });
    inputName = s.inputNames[0]; outputName = s.outputNames[0];
    const warm = new ort.Tensor('float32', new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE]);
    const o = await s.run({ [inputName]: warm });
    const ot = o[outputName] as any;
    const d: Float32Array = ot.location && ot.location !== 'cpu' ? await ot.getData(true) : ot.data;
    diag.push('wgpu ok loc=' + (ot.location || '?') + ' dlen=' + (d ? d.length : 'null'));
    if (!d || d.length === 0) throw new Error('webgpu empty');
    session = s; backend = 'onnx@webgpu';
  } catch (e) {
    diag.push('wgpu ERR ' + String(e).slice(0, 140));
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
  post({ type: 'diag', message: diag.join(' | ') + ' ||LOGS|| ' + ortLogs.slice(-6).join(' ;; ') });
  post({ type: 'ready', backend });
}

async function segment(pixels: ArrayBuffer, w: number, h: number) {
  if (!session || busy) return;
  busy = true;
  const t0 = performance.now();
  try {
    // RGBA 256² -> NCHW float [0,1] (3 плоскости R,G,B).
    const px = new Uint8ClampedArray(pixels);
    const HW = w * h;
    for (let i = 0, p = 0; i < HW; i++, p += 4) {
      inputBuf[i] = px[p] / 255;
      inputBuf[HW + i] = px[p + 1] / 255;
      inputBuf[2 * HW + i] = px[p + 2] / 255;
    }
    const tensor = new ort.Tensor('float32', inputBuf, [1, 3, h, w]);
    const out = await session.run({ [inputName]: tensor });
    const ot = out[outputName] as any;
    const hair: Float32Array = ot.location && ot.location !== 'cpu'
      ? await ot.getData(true) : ot.data; // [1,256,256,1] вероятность волос

    for (let i = 0; i < HW; i++) {
      const v = hair[i] * 255;
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
