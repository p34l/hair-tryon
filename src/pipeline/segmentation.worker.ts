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
  let fileset: any = null;

  // Лестница фолбэка. На многих Android GPU-путь МУЛЬТИКЛАССА в WebGL/воркере
  // отдаёт 0 масок при ожидаемых 6 ("confidence_mask_count 0 vs 6",
  // "norm_rect was not ok"). При сбое инференса спускаемся на ступень ниже:
  //   1) мультиклас @ GPU  — лучшая якість маски (десктоп/iOS, исправные Android);
  //   2) hair-модель @ GPU — её GPU-делегат устойчив; БЫСТРО и без CPU;
  //   3) hair-модель @ CPU — крайний случай, лишь бы не падать красным экраном.
  // GPU остаётся приоритетом: на CPU сходим только если и hair@GPU не поднялся.
  type Rung = { model: string; delegate: 'GPU' | 'CPU' };
  let ladder: Rung[] = [];
  let rung = 0;

  const offscreen = new OffscreenCanvas(INFERENCE_SIZE, INFERENCE_SIZE);
  const offCtx = offscreen.getContext('2d', { willReadFrequently: true })!;

  let maskBuffer = new Uint8Array(INFERENCE_SIZE * INFERENCE_SIZE);
  // Последний использованный timestamp segmentForVideo (VIDEO-режим требует
  // строго возрастающих меток). Прогрев занимает 0, реальные кадры идут дальше.
  let lastTs = -1;

  function post(msg: any, transfer?: Transferable[]) {
    (self as any).postMessage(msg, transfer ?? []);
  }

  // Короткая метка активной ступени для диагностики на экране.
  function backendLabel(): string {
    const c = ladder[rung];
    if (!c) return '';
    const m = c.model.indexOf('hair') >= 0 ? 'hair' : 'multi';
    return m + '@' + c.delegate;
  }

  async function createSegmenter() {
    if (!fileset) fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
    if (segmenter) {
      try { segmenter.close?.(); } catch {}
      segmenter = null;
    }
    const cfg = ladder[rung];
    segmenter = await ImageSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: cfg.model, delegate: cfg.delegate },
      runningMode: 'VIDEO',
      // Мягкая (confidence) маска класса hair — без бинарного порога (край не
      // «кипит»). categoryMask НЕ запрашиваем: её argmax — лишняя работа каждый
      // инференс (грелся GPU), а используем мы только confidenceMasks[1].
      // Обе модели (мультиклас и hair) держат hair на индексе 1.
      outputCategoryMask: false,
      outputConfidenceMasks: true,
    });
  }

  // Спуститься на следующую ступень лестницы после сбоя текущей и пересоздать
  // сегментатор. Возвращает true, если переключение произошло (есть смысл
  // повторить инференс). Когда ступени кончились — false (выше отдадим ошибку).
  async function fallbackNext(): Promise<boolean> {
    if (rung >= ladder.length - 1) return false;
    rung++;
    lastTs = -1;
    try {
      await createSegmenter();
      post({ type: 'info', message: `segmenter fallback -> ${ladder[rung].model} @ ${ladder[rung].delegate}` });
      return true;
    } catch (err) {
      // Эта ступень тоже не создалась — пробуем спуститься ещё ниже.
      return fallbackNext();
    }
  }

  async function init(modelPath: string, hairModelPath?: string) {
    // Собираем лестницу. Если hair-модель не передали — деградируем к
    // прежнему поведению (мультиклас GPU -> CPU).
    const hair = hairModelPath || modelPath;
    ladder = [
      { model: modelPath, delegate: 'GPU' },
      { model: hair, delegate: 'GPU' },
      { model: hair, delegate: 'CPU' },
    ];
    rung = 0;
    try {
      await createSegmenter();
      // Прогрев на холостом кадре: ПЕРВЫЙ инференс компилирует GPU-кернелы —
      // делаем это сейчас, на экране загрузки, чтобы при старте камеры не было
      // лага. 'ready' шлём только после прогрева (с подстраховкой по таймауту).
      warmupThenReady();
    } catch (err) {
      // Сбой создания на текущей ступени — спускаемся ниже прежде, чем сдаваться.
      if (await fallbackNext()) {
        warmupThenReady();
      } else {
        post({ type: 'error', message: String(err) });
      }
    }
  }

  function warmupThenReady() {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      post({ type: 'ready', backend: backendLabel() });
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
    } catch (err) {
      // Синхронный сбой инференса (типичный Android-кейс с мультиклас@GPU) —
      // спускаемся на ступень ниже (hair@GPU) и греемся заново.
      fallbackNext().then((ok) => {
        if (ok && !done) warmupThenReady();
        else finish();
      });
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

    const t0 = performance.now();
    try {
      segmenter.segmentForVideo(offscreen, ts, (result: any) => {
        // t1: инференс завершён (колбэк). Разница t1−t0 ≈ чистый инференс;
        // дальше идёт readback маски (getAsFloat32Array — GPU→CPU столл) + обработка.
        const t1 = performance.now();
        // Предпочитаем мягкую confidence-маску класса hair: вероятность 0..1
        // без порога. Это убирает бинарное «мерцание» на границе прядей.
        const confMasks = result.confidenceMasks;
        const hairConf = confMasks && confMasks[HAIR_CLASS];
        let wrote = false;

        if (hairConf) {
          // КРИТИЧНО для перфа: читаем маску как UINT8, а не Float32. Float32-
          // readback на мобильном WebGL идёт медленным путём (RGBA32F) и стопорит
          // общий GPU — это и был «100мс» (rd ~75мс). Uint8 — вчетверо меньше
          // данных и быстрый RGBA8-путь. Для мягкой маски качество идентично:
          // мы всё равно квантуем выход в 8 бит. confidence приходит как 0..255.
          const probs: Uint8Array = hairConf.getAsUint8Array();
          if (maskBuffer.length !== probs.length) maskBuffer = new Uint8Array(probs.length);
          // Пороги smoothstep в 0..255 (0.12*255≈31, ширина 0.76*255≈194).
          const LO = 31, INV = 1 / 194;
          for (let i = 0; i < probs.length; i++) {
            // Мягкий контраст вероятности: smoothstep гасит лишь самый
            // низкоуверенный спекл фона, но СОХРАНЯЕТ широкий градиент на переходе
            // волосы↔фон — чтобы край перекраски был мягким, а не резким (особенно
            // заметно на ярких цветах). Без бинаризации.
            let t = (probs[i] - LO) * INV;
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
        const now = performance.now();
        post(
          { type: 'mask', data: out, width: INFERENCE_SIZE, height: INFERENCE_SIZE,
            infMs: t1 - t0, readMs: now - t1, backend: backendLabel() },
          [out.buffer],
        );
      });
    } catch (err) {
      // Делегат упал на реальном кадре (Android: mask count 0 vs 6 /
      // norm_rect not ok) — спускаемся на следующую ступень (hair@GPU, затем
      // hair@CPU) и повторяем этот кадр. Ступени кончились — отдаём ошибку.
      if (rung < ladder.length - 1) {
        fallbackNext().then((ok) => {
          if (ok) segment(pixels, w, h, timestamp);
          else post({ type: 'error', message: 'segmentForVideo: ' + String(err) });
        });
      } else {
        post({ type: 'error', message: 'segmentForVideo: ' + String(err) });
      }
    }
  }

  self.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    if (msg.type === 'init') init(msg.modelPath, msg.hairModelPath);
    else if (msg.type === 'segment') segment(msg.pixels, msg.width, msg.height, msg.timestamp);
  };
})();
