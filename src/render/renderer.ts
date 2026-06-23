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
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private videoTex: WebGLTexture;
  private maskTex: WebGLTexture;
  private vao: WebGLVertexArrayObject;

  // uniform-локации
  private uVideo: WebGLUniformLocation;
  private uMask: WebGLUniformLocation;
  private uColor: WebGLUniformLocation;
  private uStrength: WebGLUniformLocation;
  private uSatScale: WebGLUniformLocation;
  private uMirror: WebGLUniformLocation;
  private uMaskTexel: WebGLUniformLocation;
  private uFeather: WebGLUniformLocation;
  private uSplit: WebGLUniformLocation;
  private uLumaShift: WebGLUniformLocation;
  private uLut: WebGLUniformLocation;
  private uLutRow: WebGLUniformLocation;
  private uColorSharp: WebGLUniformLocation;
  private uEdgeLow: WebGLUniformLocation;
  private uEdgeHigh: WebGLUniformLocation;
  private lutTex: WebGLTexture;
  private lutReady = false;
  private lutRow = -1.0; // строка текущего оттенка в LUT (0..1); <0 — HSL-фолбэк

  // текущее состояние
  private color: RGB = { r: 1, g: 0, b: 0 };
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
    const gl = createGLContext(canvas);
    this.gl = gl;
    this.program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);

    this.uVideo = gl.getUniformLocation(this.program, 'u_video')!;
    this.uMask = gl.getUniformLocation(this.program, 'u_mask')!;
    this.uColor = gl.getUniformLocation(this.program, 'u_targetColor')!;
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

    this.videoTex = createTexture(gl);
    this.maskTex = createTexture(gl);

    // LUT-атлас (яркость->цвет по оттенкам). Грузим асинхронно; до загрузки
    // u_lutRow = -1 (HSL-фолбэк).
    this.lutTex = createTexture(gl);
    const lutImg = new Image();
    lutImg.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, lutImg);
      this.lutReady = true;
    };
    lutImg.src = LUT.atlas;

    // Пустой VAO — вершины генерим из gl_VertexID, буферы не нужны.
    this.vao = gl.createVertexArray()!;

    // Инициализируем маску нулями (пока инференс не пришёл — ничего не красим).
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.R8, INFERENCE_SIZE, INFERENCE_SIZE, 0,
      gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(INFERENCE_SIZE * INFERENCE_SIZE),
    );
  }

  setColor(rgb: RGB) {
    this.color = rgb;
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
    const gl = this.gl;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
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
    this.maskReady = true;
  }

  /** Рисует один кадр. Источник — video или image (для debug-загрузки фото). */
  render(source: HTMLVideoElement | HTMLImageElement) {
    const gl = this.gl;
    const v = source as HTMLVideoElement;
    const i = source as HTMLImageElement;
    const w = v.videoWidth || i.naturalWidth || 0;
    const h = v.videoHeight || i.naturalHeight || 0;
    if (!w || !h) return;

    // Рендер-таргет держим в АСПЕКТЕ видео (sampleUv 0..1 натянут на весь канвас,
    // иной аспект растянул бы картинку), но кэпим разрешение под фактический
    // размер вывода × DPR с потолком по высоте — иначе шейдер зря считает
    // per-pixel на полном 1080p (этап 5).
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssH = (gl.canvas as HTMLCanvasElement).clientHeight || h;
    // целевая высота буфера: по CSS-высоте × DPR, но не больше видео и не больше потолка
    let targetH = Math.round(cssH * dpr) || h;
    targetH = Math.min(targetH, h, MAX_RENDER_HEIGHT);
    if (targetH < 1) targetH = h;
    const targetW = Math.max(1, Math.round(w * (targetH / h)));

    if (gl.canvas.width !== targetW || gl.canvas.height !== targetH) {
      gl.canvas.width = targetW;
      gl.canvas.height = targetH;
    }
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    // Текстура видео из текущего кадра. Storage аллоцируем один раз (или при
    // смене размера источника), дальше — texSubImage2D без переаллокации.
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    if (w !== this.videoTexW || h !== this.videoTexH) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
      this.videoTexW = w;
      this.videoTexH = h;
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
    }

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.uniform1i(this.uVideo, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    gl.uniform1i(this.uMask, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
    gl.uniform1i(this.uLut, 2);
    gl.uniform1f(this.uLutRow, this.lutReady ? this.lutRow : -1.0);

    gl.uniform3f(this.uColor, this.color.r, this.color.g, this.color.b);
    // Пока маска не пришла — strength 0, чтобы не было артефактов.
    gl.uniform1f(this.uStrength, this.maskReady ? this.strength : 0.0);
    gl.uniform1f(this.uSatScale, this.satScale);
    gl.uniform1f(this.uMirror, this.mirror);
    gl.uniform2f(this.uMaskTexel, 1 / INFERENCE_SIZE, 1 / INFERENCE_SIZE);
    gl.uniform1f(this.uFeather, FEATHER_RADIUS);
    gl.uniform1f(this.uSplit, this.split);
    gl.uniform1f(this.uLumaShift, LUMA_SHIFT);
    gl.uniform1f(this.uColorSharp, GUIDED_COLOR_SHARP);
    gl.uniform1f(this.uEdgeLow, MASK_EDGE_LOW);
    gl.uniform1f(this.uEdgeHigh, MASK_EDGE_HIGH);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
