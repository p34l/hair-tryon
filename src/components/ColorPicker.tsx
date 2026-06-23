/**
 * Выбор цвета — карусель на Swiper (как в референсе): 5 сватчей в кадре,
 * центральный = выбранный, бесконечный цикл (после последнего идёт первый),
 * свайп/драг мышью и пальцем, плавная анимация. Над каруселью — карточка
 * выбранного продукта (реальное фото упаковки + код/название + claim).
 *
 * Сватч показывает реальное фото пряди (swatchImage); если его нет — CSS-текстура
 * в цвете оттенка. Цвет передаётся как CSS-переменная --c.
 */

import { useRef, useState, type CSSProperties } from 'react';
import { Swiper, SwiperSlide } from 'swiper/react';
import { Mousewheel } from 'swiper/modules';
import type { Swiper as SwiperClass } from 'swiper';
import 'swiper/css';
import type { ColorPreset } from '../types';

interface Props {
  presets: ColorPreset[];
  selectedId: string;
  onSelect: (preset: ColorPreset) => void;
}

function escapeXml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** Сгенерированный мокап коробки — фолбэк, если реального фото нет/не загрузилось. */
function boxMockup(p: ColorPreset): string {
  const c = p.swatch ?? p.hex;
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="96" height="128" viewBox="0 0 96 128">` +
    `<rect width="96" height="128" rx="9" fill="#fff"/>` +
    `<rect width="96" height="32" rx="9" fill="#111"/><rect y="20" width="96" height="12" fill="#111"/>` +
    `<text x="48" y="14" font-family="Georgia,serif" font-size="8.5" fill="#fff" text-anchor="middle">Schwarzkopf</text>` +
    `<text x="48" y="27" font-family="Arial,sans-serif" font-size="11" font-weight="700" fill="#fff" text-anchor="middle" letter-spacing="3">LIVE</text>` +
    `<rect x="9" y="40" width="78" height="48" rx="6" fill="${c}"/>` +
    `<text x="48" y="107" font-family="Arial,sans-serif" font-size="15" font-weight="800" fill="#111" text-anchor="middle">${escapeXml(p.code)}</text>` +
    `<text x="48" y="120" font-family="Arial,sans-serif" font-size="6.5" fill="#555" text-anchor="middle">${escapeXml(p.name)}</text>` +
    `</svg>`;
  return 'data:image/svg+xml,' + encodeURIComponent(svg);
}

/** Стиль сватча: реальное фото пряди фоном, иначе цвет оттенка + текстура. */
function swatchStyle(p: ColorPreset, i = 0): CSSProperties {
  const base: CSSProperties = {
    ['--c' as any]: p.swatch ?? p.hex,
    ['--a' as any]: `${100 + (i * 37) % 80}deg`,
    ['--hl' as any]: `${25 + (i * 53) % 45}%`,
  };
  if (p.swatchImage) {
    base.backgroundImage = `url(${p.swatchImage})`;
    base.backgroundSize = 'cover';
    base.backgroundPosition = 'center';
    // не подмешиваем цвет/текстуру к реальному фото пряди
    base.backgroundColor = 'transparent';
    base.backgroundBlendMode = 'normal';
  }
  return base;
}

export function ColorPicker({ presets, selectedId, onSelect }: Props) {
  const selected = presets.find((p) => p.id === selectedId) ?? presets[0];
  const swiperRef = useRef<SwiperClass | null>(null);
  const [open, setOpen] = useState(true);
  const mock = boxMockup(selected);

  return (
    <div className="color-picker">
      {open && (
        <div className="product-card">
          <img
            className="product-box"
            src={selected.image ?? mock}
            alt={selected.name}
            onError={(e) => { if (e.currentTarget.src !== mock) e.currentTarget.src = mock; }}
          />
          <div className="product-info">
            <div className="product-name">
              {selected.code ? `${selected.code} | ` : ''}{selected.name}
            </div>
            <div className="product-subtitle">{selected.subtitle.split('|')[0].trim()}</div>
          </div>
          <button className="product-close" onClick={() => setOpen(false)} aria-label="Свернуть">×</button>
        </div>
      )}

      <Swiper
        className="swatch-swiper"
        modules={[Mousewheel]}
        slidesPerView="auto"
        centeredSlides
        loop
        grabCursor
        simulateTouch
        spaceBetween={12}
        speed={350}
        mousewheel={{ forceToAxis: true, sensitivity: 1, releaseOnEdges: false }}
        onSwiper={(sw) => { swiperRef.current = sw; }}
        onSlideChange={(sw) => {
          const p = presets[sw.realIndex];
          if (p) { onSelect(p); setOpen(true); }
        }}
      >
        {presets.map((p, i) => (
          <SwiperSlide key={p.id}>
            <button
              className={`swatch ${p.id === selectedId ? 'selected' : ''}`}
              style={swatchStyle(p, i)}
              onClick={() => swiperRef.current?.slideToLoop(i, 350)}
              aria-label={p.name}
              title={p.code ? `${p.code} | ${p.name}` : p.name}
            />
          </SwiperSlide>
        ))}
      </Swiper>
    </div>
  );
}
