/**
 * Выбор цвета — карусель на Swiper (как в референсе): 5 сватчей в кадре,
 * центральный = выбранный, бесконечный цикл, свайп/драг. Над каруселью —
 * карточка выбранного продукта (фото упаковки + код/название + claim).
 *
 * Перф:
 * - Карусель вынесена в memo-компонент SwatchSwiper, который НЕ зависит от
 *   selectedId → при смене цвета Swiper не ре-рендерится (иначе loop пересоздаёт
 *   клоны и копит DOM).
 * - Все фото (сватчи + коробки) ПРЕДзагружаются при старте (на экране загрузки)
 *   и держатся декодированными — поэтому при гортании/переключении за сессию
 *   ничего не грузится и не декодируется на лету (нет лагов), а объём фиксирован.
 */

import { memo, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
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
    base.backgroundColor = 'transparent';
    base.backgroundBlendMode = 'normal';
  }
  return base;
}

/**
 * Карусель сватчей. memo + props НЕ содержат selectedId => не ре-рендерится при
 * смене цвета. Подсветка выбранного — через CSS .swiper-slide-active.
 */
const SwatchSwiper = memo(function SwatchSwiper({
  presets,
  onSelect,
}: {
  presets: ColorPreset[];
  onSelect: (p: ColorPreset) => void;
}) {
  const swiperRef = useRef<SwiperClass | null>(null);
  // Дебаунс: при быстром гортании коммитим цвет только после остановки (~120 мс).
  const selectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const debouncedSelect = (p: ColorPreset) => {
    if (selectTimer.current) clearTimeout(selectTimer.current);
    selectTimer.current = setTimeout(() => onSelect(p), 120);
  };
  useEffect(() => () => { if (selectTimer.current) clearTimeout(selectTimer.current); }, []);

  const slides = useMemo(
    () =>
      presets.map((p, i) => (
        <SwiperSlide key={p.id}>
          <button
            className="swatch"
            style={swatchStyle(p, i)}
            onClick={() => swiperRef.current?.slideToLoop(i, 350)}
            aria-label={p.name}
            title={p.code ? `${p.code} | ${p.name}` : p.name}
          />
        </SwiperSlide>
      )),
    [presets],
  );

  return (
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
        if (p) debouncedSelect(p);
      }}
    >
      {slides}
    </Swiper>
  );
});

export const ColorPicker = memo(function ColorPicker({ presets, selectedId, onSelect }: Props) {
  const selected = presets.find((p) => p.id === selectedId) ?? presets[0];
  const [open, setOpen] = useState(true);
  const mock = boxMockup(selected);

  // ПРЕДзагрузка всех фото (сватчи + коробки) один раз при монтировании (идёт на
  // экране загрузки параллельно с моделью). Держим ссылки в ref, чтобы картинки
  // оставались декодированными — тогда при гортании/смене оттенков за сессию
  // ничего не грузится и не декодируется на лету (нет лагов). Объём фиксирован,
  // т.к. реальные утечки (createImageBitmap/SRGB) уже устранены.
  const preloadRef = useRef<HTMLImageElement[]>([]);
  // ТЕСТ ПРИЧИНЫ ЛАГА: прелоад ~150 картинок ОТКЛЮЧЁН — проверяем, не он ли держит
  // память и роняет FPS. Вернуть: раскомментировать тело эффекта.
  useEffect(() => {
    // const imgs: HTMLImageElement[] = [];
    // for (const p of presets) {
    //   if (p.swatchImage) { const im = new Image(); im.decoding = 'async'; im.src = p.swatchImage; imgs.push(im); }
    //   if (p.image) { const im = new Image(); im.decoding = 'async'; im.src = p.image; imgs.push(im); }
    // }
    // preloadRef.current = imgs;
    return () => { preloadRef.current = []; };
  }, [presets]);

  // Стабильный колбэк выбора: коммитит цвет + раскрывает карточку. Стабилен =>
  // SwatchSwiper не ре-рендерится при смене цвета.
  const handleSelect = useCallback(
    (p: ColorPreset) => { onSelect(p); setOpen(true); },
    [onSelect],
  );

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

      <SwatchSwiper presets={presets} onSelect={handleSelect} />
    </div>
  );
});
