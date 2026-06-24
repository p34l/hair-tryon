/**
 * CameraView — основной экран: скрытое <video> (источник), видимый <canvas>
 * (вывод WebGL), оверлей UI. Здесь сшивается весь пайплайн:
 *   камера/фото -> createImageBitmap(256) -> worker(сегментация)
 *               -> MaskProcessor -> Renderer(перекраска) -> canvas
 */

import { useEffect, useRef, useState } from 'react';
import { Renderer } from '../render/renderer';
import { MaskProcessor } from '../pipeline/maskProcessing';
import { startCamera } from '../pipeline/camera';
import { ColorPicker } from './ColorPicker';
import { IntensityToggle } from './IntensityToggle';
import { LegalOverlay } from './LegalOverlay';
import {
  PRESETS, INFERENCE_SIZE, INTENSITY_PARAMS, MODEL, LOGO,
  hexToRgb, CAPTURE_COUNTDOWN, FACE_DETECT_INTERVAL_MS,
} from '../config';
import type { Intensity, ColorPreset, WorkerResponse } from '../types';

type Status = 'loading' | 'ready' | 'denied' | 'error';

// --- Графические иконки (currentColor) ---
const IconCamera = () => (
  <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor"
    strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 8.5A1.5 1.5 0 0 1 5.5 7H8l1.2-1.8a1 1 0 0 1 .83-.45h3.94a1 1 0 0 1 .83.45L16 7h2.5A1.5 1.5 0 0 1 20 8.5v9A1.5 1.5 0 0 1 18.5 19h-13A1.5 1.5 0 0 1 4 17.5z" />
    <circle cx="12" cy="13" r="3.3" />
  </svg>
);
const IconCameraFlip = () => (
  <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor"
    strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3.5" y="7" width="17" height="12" rx="2.5" />
    <path d="M9.2 12.6a3 3 0 0 1 5.1-1.7" /><polyline points="14.6,8.9 14.6,11.1 12.4,11.1" />
    <path d="M14.8 13.4a3 3 0 0 1-5.1 1.7" /><polyline points="9.4,17.1 9.4,14.9 11.6,14.9" />
  </svg>
);
const IconSplit = () => (
  <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8">
    <rect x="3.5" y="5" width="17" height="14" rx="2.5" />
    <line x1="12" y1="5" x2="12" y2="19" strokeDasharray="1.8 2.4" strokeLinecap="round" />
  </svg>
);
const IconUpload = () => (
  <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor"
    strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 15V4" /><path d="M8 8l4-4 4 4" /><path d="M5 15v3.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V15" />
  </svg>
);
const IconClose = () => (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"
    strokeWidth="2" strokeLinecap="round">
    <line x1="6" y1="6" x2="18" y2="18" /><line x1="18" y1="6" x2="6" y2="18" />
  </svg>
);
const IconLegal = () => (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"
    strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 4H7a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V8z" />
    <path d="M14 4v4h4" /><path d="M9 12h6" /><path d="M9 15.5h6" />
  </svg>
);

/**
 * Снимок кадра в портретном кропе под форму карточки (9:16): вырезаем центр
 * GL-canvas (он в разрешении видео, обычно ландшафт) и отдаём data-URL.
 */
function cropPortrait(canvas: HTMLCanvasElement, preset: ColorPreset): string {
  const cw = canvas.width, ch = canvas.height;
  const aspect = 9 / 16;
  let w = ch * aspect, h = ch, x = (cw - w) / 2, y = 0;
  if (w > cw) { w = cw; h = cw / aspect; x = 0; y = (ch - h) / 2; }
  const W = Math.round(w), H = Math.round(h);
  const tmp = document.createElement('canvas');
  tmp.width = W;
  tmp.height = H;
  const ctx = tmp.getContext('2d')!;
  ctx.drawImage(canvas, x, y, w, h, 0, 0, W, H);

  // Карточка продукта снизу (цвет-сватч + текст — без CDN-картинок,
  // иначе canvas «протухнет» и toDataURL упадёт).
  const pad = Math.round(W * 0.045);
  const cardH = Math.round(W * 0.17);
  const cardY = H - cardH - pad;
  const cardX = pad, cardW = W - pad * 2;
  ctx.fillStyle = 'rgba(0,0,0,0.46)';
  ctx.beginPath();
  // roundRect есть не везде (старый Safari) — фолбэк на обычный прямоугольник.
  if (typeof (ctx as any).roundRect === 'function') {
    (ctx as any).roundRect(cardX, cardY, cardW, cardH, cardH / 2);
    ctx.fill();
  } else {
    ctx.fillRect(cardX, cardY, cardW, cardH);
  }

  const r = cardH * 0.34;
  const cx = cardX + cardH * 0.55, cy = cardY + cardH * 0.5;
  ctx.fillStyle = preset.swatch ?? preset.hex;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.6)';
  ctx.lineWidth = Math.max(1, W * 0.004);
  ctx.stroke();

  // Текст: уменьшаем шрифт, пока не влезет (без многоточия).
  const tx = cardX + cardH + pad * 0.4;
  const maxTextW = cardX + cardW - tx - pad * 0.6;
  const drawFit = (text: string, weight: string, baseSize: number, minSize: number, ty: number) => {
    let size = baseSize;
    ctx.font = `${weight} ${size}px -apple-system, BlinkMacSystemFont, sans-serif`;
    while (size > minSize && ctx.measureText(text).width > maxTextW) {
      size -= 1;
      ctx.font = `${weight} ${size}px -apple-system, BlinkMacSystemFont, sans-serif`;
    }
    ctx.fillText(text, tx, ty);
  };
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  const name = (preset.code ? `${preset.code} | ` : '') + preset.name;
  const sub = preset.subtitle.split('|')[0].trim(); // короткий подзаголовок
  ctx.fillStyle = '#fff';
  drawFit(name, '700', Math.round(cardH * 0.27), Math.round(cardH * 0.15), cy - cardH * 0.13);
  ctx.fillStyle = 'rgba(255,255,255,0.78)';
  drawFit(sub, '400', Math.round(cardH * 0.2), Math.round(cardH * 0.12), cy + cardH * 0.2);

  return tmp.toDataURL('image/jpeg', 0.92);
}

export function CameraView() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const rendererRef = useRef<Renderer | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const processorRef = useRef(new MaskProcessor());
  // FaceLandmarker теперь в ОТДЕЛЬНОМ воркере (инференс ушёл с главного потока).
  const faceWorkerRef = useRef<Worker | null>(null);
  const faceInFlightRef = useRef(false);
  const faceWatchdogRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const exclusionRef = useRef<Uint8Array | null>(null);
  const cameraHandleRef = useRef<{ stop: () => void } | null>(null);
  const guideSeenRef = useRef(false); // гайд «HOW TO SAVE» показываем 1 раз за сессию
  // Замок «кадр в обработке у воркера» + watchdog, чтобы пайплайн не залип,
  // если воркер уронил кадр (без хака (worker as any).__inFlight).
  const inFlightRef = useRef(false);
  const watchdogRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Текущий blob-URL фото (для revoke) и токен запроса (анти-гонка onload).
  const blobUrlRef = useRef<string | null>(null);
  const photoReqRef = useRef(0);
  // Таймер обратного отсчёта снимка (чистим в общем cleanup).
  const captureTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const captureBusyRef = useRef(false);
  // Pipeline маски (как после Wave 3): latest-wins coalescing + mediaTime.
  const latestSourceRef = useRef<HTMLVideoElement | HTMLImageElement | null>(null);
  const frameDirtyRef = useRef(false);
  const sendSegRef = useRef<((s: HTMLVideoElement | HTMLImageElement) => void) | null>(null);
  const mediaTimeRef = useRef(0); // mediaTime кадра из rVFC для segmentForVideo

  const [status, setStatus] = useState<Status>('loading');
  const [errorMsg, setErrorMsg] = useState('');
  const [preset, setPreset] = useState<ColorPreset>(PRESETS[0]);
  const [intensity, setIntensity] = useState<Intensity>('intense');
  const [fps, setFps] = useState(0);
  const [showDisclaimer, setShowDisclaimer] = useState(true);
  const [showCameraPopup, setShowCameraPopup] = useState(true);
  const [showLegal, setShowLegal] = useState(false);
  const [mode, setMode] = useState<'camera' | 'image'>('camera');
  const [captured, setCaptured] = useState<string | null>(null);
  const [showGuide, setShowGuide] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [captureBusy, setCaptureBusy] = useState(false); // для disabled кнопки фото
  // Исключение бороды (этап 1). По умолчанию включено; тумблер для демо.
  // Split-view (before/after): вертикальная линия по центру, левая половина без фильтра.
  const [split, setSplit] = useState(false);
  // Онбординг: пока модель грузится — экран загрузки с крутящимся лого.
  const [modelReady, setModelReady] = useState(false);   // воркер прислал 'ready'
  const [minTimePassed, setMinTimePassed] = useState(false); // минимум показа анимации
  const ready = modelReady && minTimePassed; // загрузка завершена -> просим камеру

  // refs для значений, читаемых внутри стабильного цикла рендера
  const presetRef = useRef(preset);
  const intensityRef = useRef(intensity);
  const modeRef = useRef(mode);
  const splitRef = useRef(split);
  // Позиция делителя как доля ВИДИМОЙ области сцены (0..1). Шейдеру отдаём уже
  // пересчитанный uv.x с учётом object-fit: cover (см. onFrame).
  const splitFracRef = useRef(0.5);
  const dividerRef = useRef<HTMLDivElement>(null);
  presetRef.current = preset;
  intensityRef.current = intensity;
  modeRef.current = mode;
  splitRef.current = split;

  // --- Инициализация рендерера и воркера (один раз) ---
  useEffect(() => {
    const canvas = canvasRef.current!;
    try {
      rendererRef.current = new Renderer(canvas);
    } catch (e) {
      setStatus('error');
      setErrorMsg(String(e));
      return;
    }

    // Классический воркер (без type:'module'): MediaPipe ломается в module-
    // воркере (importScripts недоступен). Файл воркера не содержит ESM-import,
    // поэтому Vite отдаёт его как классический скрипт и в dev, и в build.
    const worker = new Worker(
      new URL('../pipeline/segmentation.worker.ts', import.meta.url),
    );
    workerRef.current = worker;

    const clearInFlight = () => {
      inFlightRef.current = false;
      if (watchdogRef.current) {
        clearTimeout(watchdogRef.current);
        watchdogRef.current = null;
      }
    };

    worker.onerror = (e) => {
      clearInFlight(); // не залипаем после ошибки воркера
      setStatus('error');
      setErrorMsg('Worker: ' + (e.message || 'load/runtime error'));
    };
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === 'ready') {
        setModelReady(true); // модель сегментации загружена -> можно просить камеру
      } else if (msg.type === 'error') {
        clearInFlight();
        setStatus('error');
        setErrorMsg(msg.message ?? 'Ошибка воркера');
      } else if (msg.type === 'mask') {
        clearInFlight();
        const processed = processorRef.current.process(msg.data, {
          smooth: true, // EMA-сглаживание (этап 2)
          // вычитаем зону бороды/нижней части лица (этап 1) — всегда включено
          exclusion: exclusionRef.current,
        });
        rendererRef.current?.updateMask(processed, msg.width, msg.height);
        // Latest-wins: воркер свободен — сразу шлём свежий кадр (как в Wave 3),
        // чтобы маска была максимально актуальной (лучше держится на голове).
        if (frameDirtyRef.current && latestSourceRef.current) {
          sendSegRef.current?.(latestSourceRef.current);
        }
      }
    };

    worker.postMessage({
      type: 'init',
      modelPath: new URL(MODEL.segmenter, location.origin).href,
      wasmRoot: MODEL.wasmRoot,
    });

    return () => {
      worker.terminate();
      workerRef.current = null;
      clearInFlight();
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, []);

  // --- Инициализация FaceLandmarker во ВТОРОМ воркере (вне главного потока) ---
  // Раньше detectForVideo крутился на main (через setTimeout на iOS, где нет
  // requestIdleCallback) и блокировал рендер — главная причина просадки FPS на
  // iPhone. Теперь инференс и растеризация маски целиком в воркере.
  useEffect(() => {
    const worker = new Worker(
      new URL('../pipeline/faceLandmarks.worker.ts', import.meta.url),
    );
    faceWorkerRef.current = worker;

    const clearFaceInFlight = () => {
      faceInFlightRef.current = false;
      if (faceWatchdogRef.current) {
        clearTimeout(faceWatchdogRef.current);
        faceWatchdogRef.current = null;
      }
    };

    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === 'face-mask') {
        clearFaceInFlight();
        // msg.data — перенесённый Uint8Array (или null, если лицо не найдено).
        exclusionRef.current = msg.data ?? null;
      } else if (msg.type === 'face-error') {
        clearFaceInFlight();
        exclusionRef.current = null;
      }
      // 'face-ready' — отдельных действий не требует.
    };
    worker.onerror = () => { clearFaceInFlight(); };

    worker.postMessage({
      type: 'init',
      modelPath: new URL(MODEL.faceLandmarker, location.origin).href,
    });

    return () => {
      worker.terminate();
      faceWorkerRef.current = null;
      clearFaceInFlight();
    };
  }, []);

  // --- Минимальное время показа экрана загрузки (чтобы анимация не мигала) ---
  useEffect(() => {
    const t = setTimeout(() => setMinTimePassed(true), 1600);
    return () => clearTimeout(t);
  }, []);

  // --- Цикл рендера: крутится всегда (rAF), рисует активный источник:
  //     видео камеры, когда оно играет, иначе загруженное фото. Камера
  //     стартует отдельно (по кнопке Start), а этот цикл просто отрисовывает
  //     то, что доступно. ---
  useEffect(() => {
    const renderer = rendererRef.current;
    const video = videoRef.current!;
    if (!renderer) return;
    let cancelled = false;
    let raf = 0;
    let frameCount = 0;
    let fpsT0 = performance.now();
    let lastFaceSend = 0; // троттлинг отправки кадров в face-воркер (~4 Гц)

    // Reused-canvas для подготовки кадра 256 воркерам. Вместо createImageBitmap
    // (течёт по памяти на iOS) — drawImage+getImageData и transfer буфера пикселей.
    const frameCanvas = document.createElement('canvas');
    frameCanvas.width = INFERENCE_SIZE;
    frameCanvas.height = INFERENCE_SIZE;
    const frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true })!;
    // Геометрия как раньше: source растягивается в квадрат 256 (stretch-в-256).
    const grabPixels = (source: HTMLVideoElement | HTMLImageElement): ImageData => {
      frameCtx.drawImage(source, 0, 0, INFERENCE_SIZE, INFERENCE_SIZE);
      return frameCtx.getImageData(0, 0, INFERENCE_SIZE, INFERENCE_SIZE);
    };

    const isUsable = (s: HTMLVideoElement | HTMLImageElement) => {
      const v = s as HTMLVideoElement;
      const i = s as HTMLImageElement;
      if (v.tagName === 'VIDEO') return v.readyState >= 2 && v.videoWidth > 0;
      return i.complete && i.naturalWidth > 0;
    };

    const sendToWorker = (source: HTMLVideoElement | HTMLImageElement) => {
      const worker = workerRef.current;
      if (!worker || inFlightRef.current || !isUsable(source)) return;
      inFlightRef.current = true;
      frameDirtyRef.current = false; // этот кадр уходит в обработку
      // Таймстамп = mediaTime кадра (как в Wave 3): MediaPipe VIDEO точнее сглаживает
      // по времени контента. В фото-режиме mediaTime не растёт -> performance.now().
      const ts = modeRef.current === 'camera'
        ? Math.round(mediaTimeRef.current * 1000)
        : Math.round(performance.now());
      try {
        const id = grabPixels(source); // без createImageBitmap (не течёт на iOS)
        worker.postMessage(
          { type: 'segment', pixels: id.data.buffer, width: id.width, height: id.height, timestamp: ts },
          [id.data.buffer],
        );
        // watchdog: если ответа нет 1500 мс — снимаем замок, иначе пайплайн залипнет.
        if (watchdogRef.current) clearTimeout(watchdogRef.current);
        watchdogRef.current = setTimeout(() => { inFlightRef.current = false; }, 1500);
      } catch {
        inFlightRef.current = false;
      }
    };
    sendSegRef.current = sendToWorker; // для coalescing из worker.onmessage

    const onFrame = () => {
      const src: HTMLVideoElement | HTMLImageElement =
        modeRef.current === 'image' && imgRef.current?.complete
          ? imgRef.current
          : video;
      if (!isUsable(src)) return; // ещё нет ни кадра камеры, ни фото — нечего рисовать

      const p = presetRef.current;
      // Цвет оттенка один на оба режима; Pastel отличается силой/насыщенностью
      // (INTENSITY_PARAMS), а не отдельным hex — отдельных pastel-данных нет.
      renderer.setColor(hexToRgb(p.hex));
      const ip = INTENSITY_PARAMS[intensityRef.current];
      renderer.setIntensity(ip.strength, ip.satScale);
      // LUT отключён: фото-рампы выцветают на свету и оттенок читается неверно.
      // Используем HSL-метод с насыщенным цветом оттенка (см. шейдер).
      renderer.setLutRow(-1);
      renderer.setMirror(modeRef.current === 'camera');
      // Буфер теперь в аспекте сцены (cover делается в шейдере), поэтому output
      // uv.x == доля сцены: позиция делителя идёт напрямую, без пересчёта аспекта.
      renderer.setSplit(splitRef.current ? splitFracRef.current : -1);

      renderer.render(src);
      // Регистрируем свежий кадр для latest-wins coalescing и пробуем отправить.
      latestSourceRef.current = src;
      frameDirtyRef.current = true;
      void sendToWorker(src);

      // Кадр в face-воркер (борода) — троттлинг ~4 Гц, инференс полностью вне main.
      const fw = faceWorkerRef.current;
      const tNow = performance.now();
      if (fw && !faceInFlightRef.current && tNow - lastFaceSend >= FACE_DETECT_INTERVAL_MS) {
        lastFaceSend = tNow;
        faceInFlightRef.current = true;
        const fts = modeRef.current === 'camera'
          ? Math.round(mediaTimeRef.current * 1000)
          : Math.round(tNow);
        try {
          const id = grabPixels(src); // без createImageBitmap (не течёт на iOS)
          fw.postMessage(
            { type: 'detect', pixels: id.data.buffer, width: id.width, height: id.height, timestamp: fts },
            [id.data.buffer],
          );
          // watchdog: если воркер не ответит — снимаем замок (борода не вечный приоритет)
          if (faceWatchdogRef.current) clearTimeout(faceWatchdogRef.current);
          faceWatchdogRef.current = setTimeout(() => { faceInFlightRef.current = false; }, 2000);
        } catch {
          faceInFlightRef.current = false;
        }
      }

      // Этап 1 (exclusion-маска бороды) теперь считается в ОТДЕЛЬНОМ цикле через
      // requestIdleCallback (см. эффект ниже) — detectForVideo синхронный и тяжёлый,
      // в рендер-кадре он блокировал отрисовку. Здесь только сам рендер.

      frameCount++;
      const now = performance.now();
      if (now - fpsT0 >= 1000) {
        setFps(Math.round((frameCount * 1000) / (now - fpsT0)));
        frameCount = 0;
        fpsT0 = now;
      }
    };

    // Драйвер цикла: requestVideoFrameCallback для живой камеры (рендер ровно на
    // каждый новый кадр видео — плавно, без джиттера от рассинхрона), иначе rAF
    // (фото-режим / нет rVFC). rAF-вариант давал «дёрганость», т.к. рендер не был
    // синхронизирован с кадрами камеры.
    const hasRVFC = typeof (video as any).requestVideoFrameCallback === 'function';
    const loop = (_now?: number, meta?: { mediaTime?: number }) => {
      if (cancelled) return;
      // mediaTime кадра из rVFC — таймстамп для segmentForVideo (Wave 3).
      if (meta && typeof meta.mediaTime === 'number') mediaTimeRef.current = meta.mediaTime;
      onFrame();
      if (modeRef.current === 'camera' && hasRVFC && video.readyState >= 2 && !video.paused) {
        (video as any).requestVideoFrameCallback(loop);
      } else {
        raf = requestAnimationFrame(loop);
      }
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, []);

  // exclusion-маска бороды теперь считается во ВТОРОМ воркере: кадр отправляется
  // из onFrame (троттлинг + faceInFlightRef), результат приходит в onmessage
  // выше. На главном потоке инференса больше нет.

  // Общий cleanup при размонтировании: камера, таймер снимка, blob-URL.
  useEffect(() => () => {
    cameraHandleRef.current?.stop();
    cameraHandleRef.current = null;
    if (captureTimerRef.current) {
      clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
  }, []);

  // Перетаскивание делителя: обработчики висят на самой РУЧКЕ (а не на всей
  // сцене), с pointer capture — поэтому тапы по палитре/кнопкам и закрытие
  // делителя больше не двигают линию. Позицию ведём как долю видимой области.
  const dividerDragging = useRef(false);
  const updateDivider = (clientX: number) => {
    const cv = canvasRef.current;
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    let f = (clientX - rect.left) / rect.width;
    f = Math.max(0, Math.min(1, f));
    splitFracRef.current = f;
    if (dividerRef.current) dividerRef.current.style.left = `${f * 100}%`;
  };
  const onDividerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    dividerDragging.current = true;
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
    updateDivider(e.clientX);
  };
  const onDividerMove = (e: React.PointerEvent) => {
    if (!dividerDragging.current) return;
    updateDivider(e.clientX);
  };
  const onDividerUp = (e: React.PointerEvent) => {
    dividerDragging.current = false;
    (e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId);
  };

  // Загрузить картинку (URL/файл) как источник вместо камеры.
  const loadPhotoSrc = (src: string) => {
    const img = imgRef.current;
    if (!img) return;
    const myReq = ++photoReqRef.current; // токен против гонки onload/onerror
    // revoke предыдущего blob-URL (только blob:, не /test-model.jpg и т.п.)
    if (blobUrlRef.current && blobUrlRef.current !== src) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
    if (src.startsWith('blob:')) blobUrlRef.current = src;
    img.onload = () => {
      if (myReq !== photoReqRef.current) return; // уже запросили другое фото
      processorRef.current.reset();
      setMode('image');
    };
    img.onerror = () => {
      if (myReq !== photoReqRef.current) return;
      setStatus('error');
      setErrorMsg('Не удалось загрузить изображение.');
    };
    img.src = src;
  };

  const beginCamera = async () => {
    const video = videoRef.current;
    if (!video) return;
    // Останавливаем прежний поток перед новым стартом (иначе утечка треков).
    cameraHandleRef.current?.stop();
    cameraHandleRef.current = null;
    try {
      cameraHandleRef.current = await startCamera(video);
      setMode('camera');
      setStatus('ready');
    } catch (e: any) {
      if (e?.name === 'NotAllowedError') setStatus('denied');
      else { setStatus('error'); setErrorMsg(String(e?.message ?? e)); }
      // Камеру не дали -> показываем дефолтное тестовое фото (бородатый мужчина:
      // заодно демонстрирует исключение бороды). Свою фотку можно загрузить (⤓).
      loadPhotoSrc('/test-model.jpg');
    }
  };

  // После загрузки (модель готова + прошло мин. время анимации) автоматически
  // запрашиваем камеру — показывается ДЕФОЛТНЫЙ браузерный оверлей разрешения.
  const cameraAttempted = useRef(false);
  useEffect(() => {
    if (ready && mode === 'camera' && !cameraAttempted.current) {
      cameraAttempted.current = true;
      void beginCamera();
    }
  }, [ready, mode]);

  // --- Кнопка фото: отсчёт + фриз, с защитой от мультитапа ---
  // Снимок берём через readPixels (preserveDrawingBuffer снят): просим у рендерера
  // 2D-canvas с пикселями текущего кадра и кропаем его.
  const doCapture = async () => {
    try {
      const cap = await rendererRef.current?.requestCapture();
      if (cap) {
        setCaptured(cropPortrait(cap, presetRef.current));
        setShowGuide(!guideSeenRef.current); // гайд «HOW TO SAVE» один раз за сессию
        guideSeenRef.current = true;
      }
    } catch (err) {
      console.warn('[capture] не удалось снять кадр:', err);
    } finally {
      captureBusyRef.current = false;
      setCaptureBusy(false);
    }
  };
  const handleCapture = () => {
    // guard от повторных нажатий (фикс бага референса с накладывающимися отсчётами)
    if (captureBusyRef.current || (status !== 'ready' && mode !== 'image')) return;
    captureBusyRef.current = true;
    setCaptureBusy(true);
    let n = CAPTURE_COUNTDOWN;
    setCountdown(n);
    captureTimerRef.current = setInterval(() => {
      n -= 1;
      setCountdown(n);
      if (n <= 0) {
        if (captureTimerRef.current) {
          clearInterval(captureTimerRef.current);
          captureTimerRef.current = null;
        }
        setCountdown(0);
        void doCapture();
      }
    }, 1000);
  };

  // --- Загрузка своей фотографии вместо камеры ---
  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    loadPhotoSrc(URL.createObjectURL(file));
  };

  return (
    <div className="camera-view">
      {/* источники кадров (скрыты) */}
      <video ref={videoRef} className="hidden-source" playsInline muted />
      <img ref={imgRef} className="hidden-source" alt="" crossOrigin="anonymous" />

      {/* сцена: ограниченная по ширине вертикальная «карточка» как в референсе.
          canvas + все оверлеи позиционируются относительно неё. */}
      <div className="stage">
      {/* вывод */}
      <canvas ref={canvasRef} className="output-canvas" />

      {/* Делитель before/after: видимая линия + круглая ручка с зоной захвата.
          Drag c pointer capture — только на самой ручке. */}
      {split && (
        <div
          className="split-divider"
          ref={dividerRef}
          style={{ left: `${splitFracRef.current * 100}%` }}
          onPointerDown={onDividerDown}
          onPointerMove={onDividerMove}
          onPointerUp={onDividerUp}
          onPointerCancel={onDividerUp}
        >
          <div className="split-handle" />
        </div>
      )}

      {captured && (
        <div className="capture-screen">
          <img src={captured} alt="capture" className="capture-photo" />
          <button
            className="rail-btn capture-x"
            onClick={() => { setCaptured(null); setShowGuide(false); }}
            aria-label="Закрыть"
          ><IconClose /></button>

          {showGuide && (
            <div className="guide-backdrop" onClick={() => setShowGuide(false)}>
              <div className="guide-box" onClick={(e) => e.stopPropagation()}>
                <button className="guide-x" onClick={() => setShowGuide(false)} aria-label="Закрыть">
                  <IconClose />
                </button>
                <strong>HOW TO SAVE YOUR IMAGE</strong>
                <span>Press and hold the picture to save or share it</span>
              </div>
            </div>
          )}

        </div>
      )}

      {countdown > 0 && <div className="countdown">{countdown}</div>}

      {/* брендинг */}
      <div className="brand"><img className="brand-logo" src={LOGO.icon} alt="" />Schwarzkopf</div>

      {/* левая колонка иконок */}
      <div className="left-rail">
        <button className="rail-btn" title="Юридическая информация" onClick={() => setShowLegal(true)}>
          <IconLegal />
        </button>
        <button
          className={`rail-btn ${split ? 'active' : ''}`}
          title="Сравнение до/после (двигай линию пальцем)"
          onClick={() => {
            const next = !splitRef.current;
            splitRef.current = next;          // синхронно -> следующий кадр сразу видит off
            if (next) splitFracRef.current = 0.5;
            dividerDragging.current = false;
            setSplit(next);                   // монтирует/снимает делитель + active-класс
          }}
        >
          <IconSplit />
        </button>
        <label className="rail-btn" title="Загрузить фото (debug)">
          <IconUpload />
          <input type="file" accept="image/*" onChange={handleUpload} hidden />
        </label>
        {mode === 'image' && (
          <button className="rail-btn" title="Вернуться к камере" onClick={() => { void beginCamera(); }}>
            <IconCameraFlip />
          </button>
        )}
      </div>

      {/* FPS */}
      <div className="fps">{fps} FPS</div>

      {/* правый рейл: Intense/Pastel по центру (на уровне боковых кнопок слева) */}
      <div className="right-rail">
        <IntensityToggle value={intensity} onChange={setIntensity} />
      </div>

      {/* нижняя панель: палитра + кнопка фото */}
      <div className="bottom-bar">
        <ColorPicker presets={PRESETS} selectedId={preset.id} onSelect={setPreset} />
        <button
          className="capture-btn"
          onClick={handleCapture}
          disabled={(status !== 'ready' && mode !== 'image') || captureBusy}
        >
          <IconCamera />
        </button>
      </div>

      {showLegal && <LegalOverlay onClose={() => setShowLegal(false)} />}

      {/* камера запрещена -> попап с предложением включить камеру или загрузить фото */}
      {status === 'denied' && mode === 'camera' && showCameraPopup && (
        <div className="camera-popup">
          <span>
            For the full AR experience, please turn on your camera. Otherwise, you can also
            upload a photo (⤓).
          </span>
          <button
            className="camera-popup-close"
            onClick={() => setShowCameraPopup(false)}
            aria-label="Закрыть"
          >×</button>
        </div>
      )}
      {status === 'error' && <div className="status error">Ошибка: {errorMsg}</div>}
      </div>

      {/* Экран загрузки: белый фон, крутящееся лого по центру, дисклеймер сверху.
          Держится, пока пользователь не сделает выбор: пока модель грузится и
          пока ждём ответ на запрос камеры (status === 'loading'). Уходит, когда
          камера разрешена (ready), запрещена (denied) или ошибка. */}
      {status === 'loading' && mode === 'camera' && (
        <div className="loading-screen">
          {showDisclaimer && (
            <div className={`disclaimer ${ready ? 'leaving' : ''}`}>
              <img className="disclaimer-logo" src={LOGO.icon} alt="" />
              <div>
                <strong>Disclaimer</strong>
                <span>
                  The colour results shown are for illustrative purposes only and depend on the
                  original hair colour. Individual results may vary.
                </span>
              </div>
              <button className="disclaimer-close" onClick={() => setShowDisclaimer(false)}>×</button>
            </div>
          )}
          <div className="loading-center">
            <div className="loading-logo-wrap">
              <img className="loading-logo" src={LOGO.loading} alt="Schwarzkopf" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
