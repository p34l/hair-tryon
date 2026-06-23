/**
 * CameraView — основной экран: скрытое <video> (источник), видимый <canvas>
 * (вывод WebGL), оверлей UI. Здесь сшивается весь пайплайн:
 *   камера/фото -> createImageBitmap(256) -> worker(сегментация)
 *               -> MaskProcessor -> Renderer(перекраска) -> canvas
 */

import { useEffect, useRef, useState } from 'react';
import { Renderer } from '../render/renderer';
import { MaskProcessor } from '../pipeline/maskProcessing';
import { FaceMaskBuilder } from '../pipeline/faceLandmarks';
import { startCamera } from '../pipeline/camera';
import { ColorPicker } from './ColorPicker';
import { IntensityToggle } from './IntensityToggle';
import { LegalOverlay } from './LegalOverlay';
import {
  PRESETS, INFERENCE_SIZE, INTENSITY_PARAMS, MODEL, LOGO,
  hexToRgb, CAPTURE_COUNTDOWN,
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
  (ctx as any).roundRect(cardX, cardY, cardW, cardH, cardH / 2);
  ctx.fill();

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
  // Face Landmarker для exclusion-маски бороды (этап 1).
  const faceBuilderRef = useRef<FaceMaskBuilder | null>(null);
  const exclusionRef = useRef<Uint8Array | null>(null);
  const cameraHandleRef = useRef<{ stop: () => void } | null>(null);
  const guideSeenRef = useRef(false); // гайд «HOW TO SAVE» показываем 1 раз за сессию

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
  const splitPosRef = useRef(0.5); // позиция линии split (0..1), двигается пальцем
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

    let inFlight = false;
    worker.onerror = (e) => {
      setStatus('error');
      setErrorMsg('Worker: ' + (e.message || 'load/runtime error'));
    };
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === 'ready') {
        setModelReady(true); // модель сегментации загружена -> можно просить камеру
      } else if (msg.type === 'error') {
        setStatus('error');
        setErrorMsg(msg.message ?? 'Ошибка воркера');
      } else if (msg.type === 'mask') {
        inFlight = false;
        const processed = processorRef.current.process(msg.data, {
          smooth: true, // EMA-сглаживание (этап 2)
          // вычитаем зону бороды/нижней части лица (этап 1) — всегда включено
          exclusion: exclusionRef.current,
        });
        rendererRef.current?.updateMask(processed, msg.width, msg.height);
      }
    };

    worker.postMessage({
      type: 'init',
      modelPath: new URL(MODEL.segmenter, location.origin).href,
      wasmRoot: MODEL.wasmRoot,
    });

    // экспонируем флаг через замыкание ниже
    (worker as any).__inFlight = () => inFlight;
    (worker as any).__setInFlight = (v: boolean) => { inFlight = v; };

    return () => {
      worker.terminate();
    };
  }, []);

  // --- Инициализация Face Landmarker (этап 1, на главном потоке) ---
  useEffect(() => {
    const builder = new FaceMaskBuilder();
    (window as any).__face = { stage: 'init' };
    builder
      .init()
      .then(() => {
        faceBuilderRef.current = builder;
        (window as any).__face = { stage: 'ready' };
        console.log('[face] FaceLandmarker готов');
      })
      .catch((e) => {
        (window as any).__face = { stage: 'init-error', error: String(e) };
        console.warn('[face] init failed (борода не исключается):', e);
      });
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
    let tick = 0;
    let faceTick = 0; // монотонный timestamp для FaceLandmarker

    const isUsable = (s: HTMLVideoElement | HTMLImageElement) => {
      const v = s as HTMLVideoElement;
      const i = s as HTMLImageElement;
      if (v.tagName === 'VIDEO') return v.readyState >= 2 && v.videoWidth > 0;
      return i.complete && i.naturalWidth > 0;
    };

    const sendToWorker = async (source: HTMLVideoElement | HTMLImageElement) => {
      const worker = workerRef.current as any;
      if (!worker || worker.__inFlight() || !isUsable(source)) return;
      worker.__setInFlight(true);
      try {
        const bitmap = await createImageBitmap(source, {
          resizeWidth: INFERENCE_SIZE,
          resizeHeight: INFERENCE_SIZE,
          resizeQuality: 'low',
        });
        worker.postMessage({ type: 'segment', bitmap, timestamp: tick++ * 33 }, [bitmap]);
      } catch {
        worker.__setInFlight(false);
      }
    };

    const onFrame = () => {
      const src: HTMLVideoElement | HTMLImageElement =
        modeRef.current === 'image' && imgRef.current?.complete
          ? imgRef.current
          : video;
      if (!isUsable(src)) return; // ещё нет ни кадра камеры, ни фото — нечего рисовать

      const p = presetRef.current;
      const isPastel = intensityRef.current === 'pastel';
      // Pastel использует отдельный (реальный из AR-API) пастельный цвет.
      renderer.setColor(hexToRgb(isPastel && p.pastelHex ? p.pastelHex : p.hex));
      const ip = INTENSITY_PARAMS[intensityRef.current];
      renderer.setIntensity(ip.strength, ip.satScale);
      // LUT отключён: фото-рампы выцветают на свету и оттенок читается неверно.
      // Используем HSL-метод с насыщенным цветом оттенка (см. шейдер).
      renderer.setLutRow(-1);
      renderer.setMirror(modeRef.current === 'camera');
      renderer.setSplit(splitRef.current ? splitPosRef.current : -1);

      renderer.render(src);
      void sendToWorker(src);

      // Этап 1: пересчитываем exclusion-маску бороды через кадр (детект на
      // главном потоке — дорого делать каждый кадр). detectForVideo синхронный.
      const fb = faceBuilderRef.current;
      if (fb && frameCount % 3 === 0) {
        try {
          const ex = fb.buildJawExclusion(src, faceTick++ * 40);
          exclusionRef.current = ex;
          // [debug] раз в ~30 кадров: есть ли лицо и сколько пикселей в маске бороды
          if ((frameCount % 30) === 0) {
            let px = 0;
            if (ex) for (let i = 0; i < ex.length; i++) if (ex[i] > 0) px++;
            (window as any).__face = { stage: 'detect', face: !!ex, beardPx: px };
          }
        } catch (e) {
          exclusionRef.current = null;
          (window as any).__face = { stage: 'detect-error', error: String(e) };
        }
      }

      frameCount++; // также троттлит детект бороды (каждый 2-й кадр)
      const now = performance.now();
      if (now - fpsT0 >= 1000) {
        setFps(Math.round((frameCount * 1000) / (now - fpsT0)));
        frameCount = 0;
        fpsT0 = now;
      }
    };

    // Драйвер цикла: для живой камеры — requestVideoFrameCallback (привязка к
    // реальной частоте кадров видео, как требует бриф), иначе (фото/нет rVFC) —
    // requestAnimationFrame.
    const hasRVFC = typeof (video as any).requestVideoFrameCallback === 'function';
    const loop = () => {
      if (cancelled) return;
      onFrame();
      // Камера играет -> rVFC (привязка к частоте кадров видео). Иначе -> rAF
      // (фото-режим / камера ещё не запущена). Цикл всегда жив.
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

  // Останавливаем камеру при размонтировании.
  useEffect(() => () => cameraHandleRef.current?.stop(), []);

  // Запуск камеры по жесту пользователя (кнопка Start). Вызов getUserMedia
  // прямо в обработчике клика гарантирует промпт (важно для iOS Safari).
  // Линию split двигаем ТОЛЬКО во время активного перетаскивания (нажал->ведёшь),
  // иначе случайные pointermove у сцены сдвигали её. canvas показан через
  // object-fit: cover, поэтому позицию пальца переводим в координату кадра (uv.x).
  const splitDragging = useRef(false);
  const updateSplitPos = (e: React.PointerEvent) => {
    const el = canvasRef.current;
    if (!el || !el.width || !el.height) return;
    const rect = el.getBoundingClientRect();
    const f = (e.clientX - rect.left) / rect.width;
    const boxAspect = rect.width / rect.height;
    const fbAspect = el.width / el.height;
    const uvx = fbAspect > boxAspect ? 0.5 + (f - 0.5) * (boxAspect / fbAspect) : f;
    splitPosRef.current = Math.max(0, Math.min(1, uvx));
  };
  const onSplitDown = (e: React.PointerEvent) => {
    if (!splitRef.current) return;
    splitDragging.current = true;
    updateSplitPos(e);
  };
  const onSplitMove = (e: React.PointerEvent) => {
    if (!splitRef.current || !splitDragging.current) return;
    updateSplitPos(e);
  };
  const onSplitUp = () => { splitDragging.current = false; };

  // Загрузить картинку (URL/файл) как источник вместо камеры.
  const loadPhotoSrc = (src: string) => {
    const img = imgRef.current!;
    img.onload = () => { processorRef.current.reset(); setMode('image'); };
    img.src = src;
  };

  const beginCamera = async () => {
    const video = videoRef.current;
    if (!video) return;
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
  const captureBusy = useRef(false);
  const handleCapture = () => {
    // guard от повторных нажатий (фикс бага референса с накладывающимися отсчётами)
    if (captureBusy.current || (status !== 'ready' && mode !== 'image')) return;
    captureBusy.current = true;
    let n = CAPTURE_COUNTDOWN;
    setCountdown(n);
    const timer = setInterval(() => {
      n -= 1;
      setCountdown(n);
      if (n <= 0) {
        clearInterval(timer);
        setCountdown(0);
        setCaptured(cropPortrait(canvasRef.current!, preset));
        // гайд «HOW TO SAVE» показываем только в первый раз за сессию
        setShowGuide(!guideSeenRef.current);
        guideSeenRef.current = true;
        captureBusy.current = false;
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
      <div
        className="stage"
        onPointerDown={onSplitDown}
        onPointerMove={onSplitMove}
        onPointerUp={onSplitUp}
        onPointerLeave={onSplitUp}
        onPointerCancel={onSplitUp}
      >
      {/* вывод */}
      <canvas ref={canvasRef} className="output-canvas" />

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
            if (next) splitPosRef.current = 0.5;
            splitDragging.current = false;
            setSplit(next);                   // только для active-класса кнопки
          }}
        >
          <IconSplit />
        </button>
        <label className="rail-btn" title="Загрузить фото (debug)">
          <IconUpload />
          <input type="file" accept="image/*" onChange={handleUpload} hidden />
        </label>
        {mode === 'image' && (
          <button className="rail-btn" title="Вернуться к камере" onClick={() => { setMode('camera'); void beginCamera(); }}>
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
          disabled={(status !== 'ready' && mode !== 'image') || captureBusy.current}
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
