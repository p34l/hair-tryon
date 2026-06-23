/**
 * Переключатель Intense / Pastel. Управляет одним значением strength
 * (и насыщенностью) через шейдер — без смены модели или шейдера.
 */

import type { Intensity } from '../types';

interface Props {
  value: Intensity;
  onChange: (v: Intensity) => void;
}

export function IntensityToggle({ value, onChange }: Props) {
  return (
    <div className={`intensity-toggle ${value === 'pastel' ? 'is-pastel' : 'is-intense'}`}>
      {/* скользящий бегунок-подсветка активного режима */}
      <span className="intensity-thumb" />
      {(['intense', 'pastel'] as const).map((mode) => (
        <button
          key={mode}
          className={value === mode ? 'active' : ''}
          onClick={() => onChange(mode)}
        >
          {mode === 'intense' ? 'Intense' : 'Pastel'}
        </button>
      ))}
    </div>
  );
}
