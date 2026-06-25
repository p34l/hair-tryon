/**
 * Рендер-цикл: на каждый кадр загружает текстуру видео и текстуру маски,
 * выставляет uniforms (цвет, сила, насыщенность, зеркало) и рисует
 * полноэкранный треугольник с recolor-шейдером.
 *
 * Рендеринг идёт на ПОЛНОМ разрешении дисплея — это дёшево (per-pixel),
 * а маска 256x256 апскейлится билинейно при сэмплинге.
 */

import { createGLContext, createProgram, createTexture } from './glContext';
import { VERTEX_SHADER, FRAGMENT_SHADER } from './recolorShader';
import {
  INFERENCE_SIZE, FEATHER_RADIUS, LUMA_SHIFT, LUT, MAX_RENDER_HEIGHT,
  GUIDED_COLOR_SHARP, MASK_EDGE_LOW, MASK_EDGE_HIGH,
} from '../config';
import type { RGB } from '../types';

export class Renderer {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  // GL-ресурсы создаются в initGLResources() (вызывается из конструктора и при
  // восстановлении контекста), поэтому definite-assignment (!).
  private program!: WebGLProgram;
  private videoTex!: WebGLTexture;
  private maskTex!: WebGLTexture;
  private lutTex!: WebGLTexture;
  private vao!: WebGLVertexArrayObject;

  // uniform-локации
  private uVideo!: WebGLUniformLocation;
  private uMask!: WebGLUniformLocation;
  private uTargetLab!: WebGLUniformLocation;
  private uStrength!: WebGLUniformLocation;
  private uSatScale!: WebGLUniformLocation;
  private uMirror!: WebGLUniformLocation;
  private uMaskTexel!: WebGLUniformLocation;
  private uFeather!: WebGLUniformLocation;
  private uSplit!: WebGLUniformLocation;
  private uLumaShift!: WebGLUniformLocation;
  private uLut!: WebGLUniformLocation;
  private uLutRow!: WebGLUniformLocation;
  private uColorSharp!: WebGLUniformLocation;
  private uEdgeLow!: WebGLUniformLocation;
  private uEdgeHigh!: WebGLUniformLocation;
  private uCoverScale!: WebGLUniformLocation;
  private lutImg: HTMLImageElement | null = null;
  private lutReady = false;
  private lutRow = -1.0; // строка текущего оттенка в LUT (0..1); <0 — HSL-фолбэк

  // Размер CSS-бокса канваса (кэшируем через ResizeObserver — не читаем
  // clientWidth/Height в кадре, это форс-релейаут). Рендерим буфер под видимую
  // область × DPR, а не под разрешение видео: считаем только видимые пиксели.
  private clientW = 0;
  private clientH = 0;
  private resizeObserver: ResizeObserver | null = null;

  // Снимок кадра через readPixels (без preserveDrawingBuffer): запрос ставит
  // resolver, render() после drawArrays делает readback и резолвит промис.
  private captureResolve: ((c: HTMLCanvasElement | null) => void) | null = null;
  private captureCanvas: HTMLCanvasElement | null = null;

  // Контекст потерян (вкладка/драйвер) — render/updateMask становятся no-op,
  // пока не придёт webglcontextrestored. Чёрного экрана/крэша не будет.
  private contextLost = false;

  // текущее состояние (переживает потерю контекста — не пересоздаётся)
  // Цвет цели в OKLab (считаем на CPU в setColor — чтобы не делать pow на каждый
  // пиксель в шейдере). Дефолт — OKLab красного.
  private targetLab: [number, number, number] = [0.628, 0.225, 0.126];
  private strength = 0.85;
  private satScale = 1.0;
  private mirror = 1.0;
  private split = -1.0; // split-view выключен по умолчанию
  private maskReady = false;

  // Размеры уже выделенного storage текстур (этап 4). Storage аллоцируем один
  // раз через texImage2D, дальше обновляем содержимое через texSubImage2D —
  // переаллокация на каждый кадр (особенно 1080p видео) бьёт по FPS.
  private videoTexW = 0;
  private videoTexH = 0;
  private maskTexW = INFERENCE_SIZE;
  private maskTexH = INFERENCE_SIZE;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = createGLContext(canvas);
    this.gl = gl;
    canvas.addEventListener('webglcontextlost', this.handleContextLost);
    canvas.addEventListener('webglcontextrestored', this.handleContextRestored);

    // Кэшируем размер CSS-бокса канваса (без чтения clientWidth/Height в кадре).
    this.clientW = canvas.clientWidth;
    this.clientH = canvas.clientHeight;
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver((entries) => {
        const r = entries[0]?.contentRect;
        if (r) { this.clientW = r.width; this.clientH = r.height; }
      });
      this.resizeObserver.observe(canvas);
    }
    if (gl.isContextLost()) {
      // Контекст уже потерян — типично при StrictMode-remount на ТОМ ЖЕ canvas,
      // если предыдущий dispose() вызвал loseContext(). getContext возвращает тот
      // же мёртвый контекст, поэтому просим восстановление; ресурсы создадутся в
      // webglcontextrestored, а render/updateMask пока no-op (contextLost=true).
      this.contextLost = true;
      gl.getExtension('WEBGL_lose_context')?.restoreContext();
    } else {
      this.initGLResources();
    }
  }

  /**
   * Создаёт (или пересоздаёт после restore) все GL-ресурсы: программу, локации
   * uniform-ов, текстуры, VAO, начальную маску и LUT. Состояние (цвет/сила/…)
   * хранится отдельно и не сбрасывается.
   */
  private initGLResources() {
    const gl = this.gl;
    this.program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);

    this.uVideo = gl.getUniformLocation(this.program, 'u_video')!;
    this.uMask = gl.getUniformLocation(this.program, 'u_mask')!;
    this.uTargetLab = gl.getUniformLocation(this.program, 'u_targetLab')!;
    this.uStrength = gl.getUniformLocation(this.program, 'u_strength')!;
    this.uSatScale = gl.getUniformLocation(this.program, 'u_satScale')!;
    this.uMirror = gl.getUniformLocation(this.program, 'u_mirror')!;
    this.uMaskTexel = gl.getUniformLocation(this.program, 'u_maskTexel')!;
    this.uFeather = gl.getUniformLocation(this.program, 'u_feather')!;
    this.uSplit = gl.getUniformLocation(this.program, 'u_split')!;
    this.uLumaShift = gl.getUniformLocation(this.program, 'u_lumaShift')!;
    this.uLut = gl.getUniformLocation(this.program, 'u_lut')!;
    this.uLutRow = gl.getUniformLocation(this.program, 'u_lutRow')!;
    this.uColorSharp = gl.getUniformLocation(this.program, 'u_colorSharp')!;
    this.uEdgeLow = gl.getUniformLocation(this.program, 'u_edgeLow')!;
    this.uEdgeHigh = gl.getUniformLocation(this.program, 'u_edgeHigh')!;
    this.uCoverScale = gl.getUniformLocation(this.program, 'u_coverScale')!;

    this.videoTex = createTexture(gl);
    this.maskTex = createTexture(gl);
    this.lutTex = createTexture(gl);

    // Размеры storage сброшены — следующий кадр заново аллоцирует текстуры.
    this.videoTexW = 0;
    this.videoTexH = 0;
    this.maskTexW = INFERENCE_SIZE;
    this.maskTexH = INFERENCE_SIZE;
    this.maskReady = false;
    this.lutReady = false;

    // Пустой VAO — вершины генерим из gl_VertexID, буферы не нужны.
    this.vao = gl.createVertexArray()!;

    // --- Статический GL-стейт (один раз, не в кадре) ---
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);

    // Сэмплеры жёстко привязаны к юнитам: video=0, mask=1, lut=2. Каждую текстуру
    // держим на своём юните; апдейты (render/updateMask/loadLut) выбирают свой
    // activeTexture, поэтому повторно uniform1i/перепривязки в кадре не нужны.
    gl.uniform1i(this.uVideo, 0);
    gl.uniform1i(this.uMask, 1);
    gl.uniform1i(this.uLut, 2);

    // Константные uniform-ы (из config) — тоже один раз.
    gl.uniform2f(this.uMaskTexel, 1 / INFERENCE_SIZE, 1 / INFERENCE_SIZE);
    gl.uniform1f(this.uFeather, FEATHER_RADIUS);
    gl.uniform1f(this.uLumaShift, LUMA_SHIFT);
    gl.uniform1f(this.uColorSharp, GUIDED_COLOR_SHARP);
    gl.uniform1f(this.uEdgeLow, MASK_EDGE_LOW);
    gl.uniform1f(this.uEdgeHigh, MASK_EDGE_HIGH);

    // Маску инициализируем нулями на юните 1.
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.R8, INFERENCE_SIZE, INFERENCE_SIZE, 0,
      gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(INFERENCE_SIZE * INFERENCE_SIZE),
    );

    // LUT на юните 2.
    this.loadLut();

    // Видео-текстуру держим на юните 0 и оставляем его активным.
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
  }

  /** Грузит/перезаливает LUT-атлас. После restore переиспользует кэш-картинку. */
  private loadLut() {
    const gl = this.gl;
    const upload = (img: HTMLImageElement) => {
      if (this.lutImg !== img || this.contextLost) return; // устарел / контекст потерян
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.activeTexture(gl.TEXTURE0); // возвращаем активный юнit к видео
      this.lutReady = true;
    };
    // Если картинка уже скачана (например после restore) — заливаем сразу.
    if (this.lutImg && this.lutImg.complete && this.lutImg.naturalWidth > 0) {
      upload(this.lutImg);
      return;
    }
    const img = new Image();
    this.lutImg = img;
    img.onload = () => upload(img);
    img.src = LUT.atlas;
  }

  private handleContextLost = (e: Event) => {
    // preventDefault обязателен, иначе контекст не будет восстановлен.
    e.preventDefault();
    this.contextLost = true;
  };

  private handleContextRestored = () => {
    this.contextLost = false;
    this.initGLResources();
  };

  /** Освобождает все GL-ресурсы и слушатели (вызывать при unmount). */
  dispose() {
    const gl = this.gl;
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    if (this.captureResolve) { this.captureResolve(null); this.captureResolve = null; }
    this.canvas.removeEventListener('webglcontextlost', this.handleContextLost);
    this.canvas.removeEventListener('webglcontextrestored', this.handleContextRestored);
    if (this.lutImg) {
      this.lutImg.onload = null;
      this.lutImg = null;
    }
    gl.deleteTexture(this.videoTex);
    gl.deleteTexture(this.maskTex);
    gl.deleteTexture(this.lutTex);
    gl.deleteProgram(this.program);
    gl.deleteVertexArray(this.vao);
    // Освобождаем сам контекст (после снятия слушателей — событие не обработаем).
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  }

  /** Цель приходит в sRGB 0..1; OKLab считаем здесь (на CPU), а не per-pixel. */
  setColor(rgb: RGB) {
    // sRGB -> linear (pow 2.2 — согласовано с srgb2lin в шейдере) -> OKLab.
    const R = Math.pow(Math.max(rgb.r, 0), 2.2);
    const G = Math.pow(Math.max(rgb.g, 0), 2.2);
    const B = Math.pow(Math.max(rgb.b, 0), 2.2);
    const l = 0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B;
    const m = 0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B;
    const s = 0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B;
    const l_ = Math.cbrt(l);
    const m_ = Math.cbrt(m);
    const s_ = Math.cbrt(s);
    this.targetLab[0] = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    this.targetLab[1] = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    this.targetLab[2] = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;
  }

  /** Выбрать строку LUT для текущего оттенка (0..1) или -1, чтобы отключить LUT. */
  setLutRow(row01: number) {
    this.lutRow = row01;
  }

  /** strength + satScale задаются переключателем Intense/Pastel. */
  setIntensity(strength: number, satScale: number) {
    this.strength = strength;
    this.satScale = satScale;
  }

  setMirror(on: boolean) {
    this.mirror = on ? 1.0 : 0.0;
  }

  /** Split-view: позиция линии (0..1) или <0 чтобы выключить. */
  setSplit(pos: number) {
    this.split = pos;
  }

  /** Загружает новую маску волос (Uint8, INFERENCE_SIZE²) в текстуру. */
  updateMask(data: Uint8Array, width: number, height: number) {
    if (this.contextLost) return;
    const gl = this.gl;
    // Маска живёт на текстурном юните 1 — выбираем его, не трогая юнит видео (0).
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    if (width !== this.maskTexW || height !== this.maskTexH) {
      // Размер изменился (или первый кадр) — (пере)аллоцируем storage.
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.R8, width, height, 0,
        gl.RED, gl.UNSIGNED_BYTE, data,
      );
      this.maskTexW = width;
      this.maskTexH = height;
    } else {
      // Storage уже есть — обновляем содержимое без переаллокации.
      gl.texSubImage2D(
        gl.TEXTURE_2D, 0, 0, 0, width, height,
        gl.RED, gl.UNSIGNED_BYTE, data,
      );
    }
    gl.activeTexture(gl.TEXTURE0); // возвращаем активный юнit к видео
    this.maskReady = true;
  }

  /**
   * Запрашивает снимок текущего кадра. Резолвится 2D-canvas с пикселями буфера
   * (без preserveDrawingBuffer — readback делаем сразу после drawArrays).
   * Вернёт null, если контекст потерян.
   */
  requestCapture(): Promise<HTMLCanvasElement | null> {
    if (this.contextLost) return Promise.resolve(null);
    return new Promise((resolve) => { this.captureResolve = resolve; });
  }

  /** Читает буфер через readPixels и кладёт во flip-Y 2D-canvas. */
  private readToCanvas(): HTMLCanvasElement {
    const gl = this.gl;
    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);

    const out = this.captureCanvas ?? document.createElement('canvas');
    this.captureCanvas = out;
    out.width = w;
    out.height = h;
    const ctx = out.getContext('2d')!;
    const img = ctx.createImageData(w, h);
    const row = w * 4;
    // readPixels отдаёт строки снизу вверх — переворачиваем по Y.
    for (let y = 0; y < h; y++) {
      const src = (h - 1 - y) * row;
      img.data.set(px.subarray(src, src + row), y * row);
    }
    ctx.putImageData(img, 0, 0);
    return out;
  }

  /**
   * Рисует один кадр. Источник — video или image (для debug-загрузки фото).
   *
   * ext — ВНЕШНЯЯ текстура маски (из MPMask.getAsWebGLTexture), которая живёт на
   * GPU в ЭТОМ ЖЕ контексте (zero-readback). Если передана — сэмплируем её как
   * u_mask напрямую, без CPU-upload. Валидна только в колбэке segmentForVideo,
   * поэтому render(ext) вызывается оттуда же. Без ext — используем свою maskTex.
   *
   * ВАЖНО: сегментатор делит этот GL-контекст и оставляет своё состояние (program,
   * FBO, привязки текстур, viewport). Поэтому в начале кадра ПОЛНОСТЬЮ
   * восстанавливаем наш стейт — иначе рисуется чёрное/мусор.
   */
  render(
    source: HTMLVideoElement | HTMLImageElement,
    ext?: { tex: WebGLTexture; w: number; h: number },
  ) {
    if (this.contextLost) return;
    const gl = this.gl;
    const v = source as HTMLVideoElement;
    const i = source as HTMLImageElement;
    const w = v.videoWidth || i.naturalWidth || 0;
    const h = v.videoHeight || i.naturalHeight || 0;
    if (!w || !h) return;

    // --- Восстановление нашего GL-стейта после MediaPipe (общий контекст) ---
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); // рисуем на экран, а не в чужой FBO
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.uniform1i(this.uVideo, 0);
    gl.uniform1i(this.uMask, 1);
    gl.uniform1i(this.uLut, 2);

    // Рендер-таргет = ВИДИМАЯ область сцены × DPR (а не разрешение видео): не
    // считаем per-pixel за пределами кропа. Аспект буфера = аспекту CSS-бокса,
    // object-fit: cover делается в шейдере (u_coverScale). Кэп по высоте.
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let bufW: number;
    let bufH: number;
    if (this.clientW > 0 && this.clientH > 0) {
      const physH = this.clientH * dpr;
      const scale = Math.min(1, MAX_RENDER_HEIGHT / physH);
      bufH = Math.max(1, Math.round(physH * scale));
      bufW = Math.max(1, Math.round(this.clientW * dpr * scale));
    } else {
      // до первого layout — фолбэк на аспект видео с кэпом по высоте
      bufH = Math.min(h, MAX_RENDER_HEIGHT);
      bufW = Math.max(1, Math.round(w * (bufH / h)));
    }

    if (gl.canvas.width !== bufW || gl.canvas.height !== bufH) {
      gl.canvas.width = bufW;
      gl.canvas.height = bufH;
    }
    gl.viewport(0, 0, bufW, bufH);

    // object-fit: cover в UV — масштаб вокруг центра, чтобы видео покрыло буфер
    // без растяжения (кропаем по большей стороне источника).
    const outAspect = bufW / bufH;
    const srcAspect = w / h;
    let coverX = 1;
    let coverY = 1;
    if (srcAspect > outAspect) coverX = outAspect / srcAspect;
    else coverY = srcAspect / outAspect;

    // Видео-текстура на юните 0. Storage один раз, далее texSubImage2D.
    // ВАЖНО: обычный RGBA8 (НЕ SRGB8_ALPHA8). На iOS Safari покадровый upload в
    // sRGB-текстуру подтекает GPU-памятью → за ~20с упирается в лимит вкладки и
    // FPS падает без отката. sRGB->linear делаем в шейдере (srgb2lin) — цвет тот же.
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    if (w !== this.videoTexW || h !== this.videoTexH) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
      this.videoTexW = w;
      this.videoTexH = h;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
    }

    // --- Маска на юните 1 ---
    gl.activeTexture(gl.TEXTURE1);
    if (ext) {
      // Внешняя GPU-текстура маски (zero-readback). Размер маски = размер входа
      // сегментатора (обычно разрешение видео). u_maskTexel/feather масштабируем,
      // чтобы окрестность joint-bilateral по UV совпала с настройкой под сетку 256.
      gl.bindTexture(gl.TEXTURE_2D, ext.tex);
      // MediaPipe-текстура маски может быть float и не фильтроваться LINEAR на
      // мобильных — ставим NEAREST/CLAMP, безопасно для сэмплинга со смещениями.
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const mw = ext.w || INFERENCE_SIZE;
      const mh = ext.h || INFERENCE_SIZE;
      gl.uniform2f(this.uMaskTexel, 1 / mw, 1 / mh);
      gl.uniform1f(this.uFeather, FEATHER_RADIUS * Math.max(1, mw / INFERENCE_SIZE));
      this.maskReady = true;
    } else {
      // Легаси-путь (фото-режим/фолбэк): наша загруженная R8-маска 256².
      gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
      gl.uniform2f(this.uMaskTexel, 1 / INFERENCE_SIZE, 1 / INFERENCE_SIZE);
      gl.uniform1f(this.uFeather, FEATHER_RADIUS);
    }

    // LUT на юните 2 (перепривязываем — MediaPipe мог сбить привязку).
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex);

    // Только меняющиеся uniform-ы.
    gl.uniform1f(this.uLutRow, this.lutReady ? this.lutRow : -1.0);
    gl.uniform3f(this.uTargetLab, this.targetLab[0], this.targetLab[1], this.targetLab[2]);
    gl.uniform1f(this.uStrength, this.maskReady ? this.strength : 0.0);
    gl.uniform1f(this.uSatScale, this.satScale);
    gl.uniform1f(this.uMirror, this.mirror);
    gl.uniform1f(this.uSplit, this.split);
    gl.uniform2f(this.uCoverScale, coverX, coverY);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Снимок (если запрошен) — пока буфер ещё не очищен композитингом.
    if (this.captureResolve) {
      const resolve = this.captureResolve;
      this.captureResolve = null;
      resolve(this.readToCanvas());
    }
  }
}
