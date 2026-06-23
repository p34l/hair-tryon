/**
 * Работа с камерой: запрос потока, привязка к видеоэлементу и драйвер
 * покадрового цикла через requestVideoFrameCallback (с фолбэком на rAF).
 */

export interface CameraHandle {
  video: HTMLVideoElement;
  stop: () => void;
}

/**
 * Запрашивает фронтальную камеру и проигрывает поток в скрытом <video>.
 * Учитывает iOS Safari: playsInline + muted + явный play() после user gesture.
 */
export async function startCamera(video: HTMLVideoElement): Promise<CameraHandle> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Камера недоступна в этом браузере (нет getUserMedia).');
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: 'user',
      // Видео (не инференс) в высоком разрешении — картинка чётче. Рендер идёт
      // на полном разрешении, инференс всё равно на 256 (бриф соблюдён).
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      // просим 60 к/с; если камера умеет только 30 — останется 30 (rVFC привязан
      // к реальной частоте кадров видео, поэтому FPS = частоте камеры).
      frameRate: { ideal: 60 },
    },
  });

  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  // iOS требует, чтобы атрибут реально стоял в DOM.
  video.setAttribute('playsinline', '');
  video.setAttribute('muted', '');

  await new Promise<void>((resolve) => {
    if (video.readyState >= 2) return resolve();
    video.onloadedmetadata = () => resolve();
  });
  await video.play();

  return {
    video,
    stop: () => {
      stream.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    },
  };
}

/** Поддерживает ли браузер requestVideoFrameCallback. */
function hasRVFC(video: HTMLVideoElement): boolean {
  return typeof (video as any).requestVideoFrameCallback === 'function';
}

/**
 * Запускает покадровый цикл, привязанный к реальной частоте кадров видео.
 * Возвращает функцию остановки. callback получает текущий timestamp (мс).
 */
export function driveFrames(
  video: HTMLVideoElement,
  callback: (timestampMs: number) => void,
): () => void {
  let stopped = false;

  if (hasRVFC(video)) {
    const loop = (_now: number, meta: { mediaTime: number }) => {
      if (stopped) return;
      callback(meta.mediaTime * 1000);
      (video as any).requestVideoFrameCallback(loop);
    };
    (video as any).requestVideoFrameCallback(loop);
  } else {
    // Фолбэк для старых браузеров (напр. старый Firefox).
    const loop = () => {
      if (stopped) return;
      callback(performance.now());
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  return () => {
    stopped = true;
  };
}
