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
      // 720p — чёткая картинка. Инференс всё равно на 256, рендер кэпится по высоте.
      width: { ideal: 1280 },
      height: { ideal: 720 },
      // frameRate.min КРИТИЧЕН на Android: без нижней границы сенсор в комнатном
      // свете (особенно при движении/тёмной сцене) удлиняет ВЫДЕРЖКУ, чтобы
      // вытянуть яркость, а длинная выдержка физически режет FPS (1/10с → 10 к/с —
      // ровно симптом «20→10 при движении головой»). Требуя min, заставляем камеру
      // держать частоту и компенсировать яркость усилением (ISO/gain), а не временем.
      frameRate: { min: 24, ideal: 30 },
    },
  });

  // Подстраховка: на части Android начальные constraints игнорируются, но
  // applyConstraints на живом треке honor'ится. Просим держать кадровую частоту;
  // ошибки/неподдержку молча глотаем (тогда остаётся поведение по умолчанию).
  const track = stream.getVideoTracks()[0];
  if (track) {
    try {
      await track.applyConstraints({ frameRate: { min: 24, ideal: 30 } } as MediaTrackConstraints);
    } catch {
      /* устройство не даёт менять frameRate на лету — ок */
    }
  }

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
