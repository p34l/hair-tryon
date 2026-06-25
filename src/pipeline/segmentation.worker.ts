/**
 * Web Worker: сегментация волос через ONNX Runtime Web на ЧИСТО-СВЁРТОЧНОЙ BiSeNet
 * (face-parsing.PyTorch), сконвертированной в ONNX.
 *
 * Почему именно эта модель: предыдущая (selfie_multiclass из tflite) — гибрид
 * CNN+attention с десятками Transpose (NHWC), которые WebGPU-EP тянет плохо/неверно
 * → инференс ~120мс и кривая маска. BiSeNet — pure CNN (НОЛЬ Transpose, NCHW),
 * WebGPU гоняет её правильно и быстро. И никакого MediaPipe → нет iOS-утечки.
 *
 * Модель: вход [1,3,512,512] NCHW, ImageNet-нормализация; в граф добавлен softmax
 * по каналам + выбор класса hair(17) -> выход [1,1,512,512] вероятность волос.
 * Маску ужимаем до 256² (под настройку recolor-шейдера) и отдаём как Uint8.
 */

import * as ort from 'onnxruntime-web';

const IN = 512;   // вход модели
const OUT = 256;  // размер маски, который ждёт пайплайн (recolor тюнен под 256)
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

ort.env.wasm.wasmPaths = '/ort/';
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'warning';
const ortLogs: string[] = [];
const _warn = console.warn.bind(console);
const _err = console.error.bind(console);
console.warn = (...a: any[]) => { ortLogs.push('W ' + a.map(String).join(' ').slice(0, 140)); _warn(...a); };
console.error = (...a: any[]) => { ortLogs.push('E ' + a.map(String).join(' ').slice(0, 140)); _err(...a); };

let session: ort.InferenceSession | null = null;
let inputName = 'input';
let outputName = 'hair';
let backend = 'onnx';
let busy = false;

// Канвасы для ресайза входного кадра до 512² (putImageData не масштабирует).
const srcCanvas = new OffscreenCanvas(OUT, OUT);
const srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true })!;
const inCanvas = new OffscreenCanvas(IN, IN);
const inCtx = inCanvas.getContext('2d', { willReadFrequently: true })!;

const inputBuf = new Float32Array(3 * IN * IN);
let maskBuf = new Uint8Array(OUT * OUT);

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
    const warm = new ort.Tensor('float32', new Float32Array(3 * IN * IN), [1, 3, IN, IN]);
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
    // Входные пиксели (обычно 256² RGBA) -> апскейл до 512² через 2 канваса.
    const src = new ImageData(new Uint8ClampedArray(pixels), w, h);
    if (srcCanvas.width !== w || srcCanvas.height !== h) { srcCanvas.width = w; srcCanvas.height = h; }
    srcCtx.putImageData(src, 0, 0);
    inCtx.drawImage(srcCanvas, 0, 0, IN, IN);
    const id = inCtx.getImageData(0, 0, IN, IN).data; // RGBA 512²

    // NCHW + ImageNet-нормализация.
    const HW = IN * IN;
    for (let i = 0, p = 0; i < HW; i++, p += 4) {
      inputBuf[i] = (id[p] / 255 - MEAN[0]) / STD[0];
      inputBuf[HW + i] = (id[p + 1] / 255 - MEAN[1]) / STD[1];
      inputBuf[2 * HW + i] = (id[p + 2] / 255 - MEAN[2]) / STD[2];
    }
    const tensor = new ort.Tensor('float32', inputBuf, [1, 3, IN, IN]);
    const out = await session.run({ [inputName]: tensor });
    const ot = out[outputName] as any;
    const hair: Float32Array = ot.location && ot.location !== 'cpu'
      ? await ot.getData(true) : ot.data; // [1,1,512,512] вероятность волос

    // Ужимаем 512² -> 256² усреднением 2x2, в Uint8.
    const ratio = IN / OUT; // 2
    for (let y = 0; y < OUT; y++) {
      for (let x = 0; x < OUT; x++) {
        const sy = y * ratio, sx = x * ratio;
        const a = hair[sy * IN + sx], b = hair[sy * IN + sx + 1];
        const c = hair[(sy + 1) * IN + sx], dd = hair[(sy + 1) * IN + sx + 1];
        const v = (a + b + c + dd) * 0.25 * 255;
        maskBuf[y * OUT + x] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
    const data = maskBuf.slice();
    post(
      { type: 'mask', data, width: OUT, height: OUT, infMs: performance.now() - t0, backend },
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
