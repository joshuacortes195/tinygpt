import { DiceFiveIcon } from '@phosphor-icons/react'
import { useId } from 'react'
import type { SamplingSettings } from '../lib/types'

interface Props {
  settings: SamplingSettings
  onChange: (settings: SamplingSettings) => void
  disabled?: boolean
}

// the sliders that control how the model picks each token
export function SettingsPanel({ settings, onChange, disabled }: Props) {
  const seedId = useId()
  const set = (patch: Partial<SamplingSettings>) => onChange({ ...settings, ...patch })

  return (
    <div className="grid gap-5">
      <Slider
        label="Temperature"
        hint="Higher is more random. 0 always picks the top guess."
        value={settings.temperature}
        min={0}
        max={1.5}
        step={0.05}
        format={(v) => v.toFixed(2)}
        onChange={(temperature) => set({ temperature })}
        disabled={disabled}
      />
      <Slider
        label="Top-k"
        hint="Only the k most likely tokens can be picked. 0 turns it off."
        value={settings.topK}
        min={0}
        max={200}
        step={1}
        format={(v) => (v === 0 ? 'off' : String(v))}
        onChange={(topK) => set({ topK })}
        disabled={disabled}
      />
      <Slider
        label="Top-p"
        hint="Keeps the smallest group of tokens that covers this much probability."
        value={settings.topP}
        min={0.05}
        max={1}
        step={0.05}
        format={(v) => (v >= 1 ? 'off' : v.toFixed(2))}
        onChange={(topP) => set({ topP })}
        disabled={disabled}
      />
      <Slider
        label="Max tokens"
        hint="How much the model writes before it stops."
        value={settings.maxTokens}
        min={10}
        max={400}
        step={10}
        format={(v) => String(v)}
        onChange={(maxTokens) => set({ maxTokens })}
        disabled={disabled}
      />

      {/* same seed and settings give the same story every time */}
      <div className="grid gap-2">
        <label htmlFor={seedId} className="label">
          Seed
        </label>
        <div className="flex gap-2">
          <input
            id={seedId}
            type="number"
            inputMode="numeric"
            min={0}
            className="field num min-w-0 flex-1"
            value={settings.seed}
            disabled={disabled}
            onChange={(e) => set({ seed: Math.max(0, Math.floor(Number(e.target.value) || 0)) })}
          />
          <button
            type="button"
            className="btn w-11 shrink-0 px-0"
            aria-label="Pick a random seed"
            disabled={disabled}
            onClick={() => set({ seed: Math.floor(Math.random() * 100000) })}
          >
            <DiceFiveIcon size={18} />
          </button>
        </div>
        <p className="text-xs leading-relaxed text-muted">The same seed and settings always give the same story.</p>
      </div>
    </div>
  )
}

interface SliderProps {
  label: string
  hint: string
  value: number
  min: number
  max: number
  step: number
  format: (value: number) => string
  onChange: (value: number) => void
  disabled?: boolean
}

function Slider({ label, hint, value, min, max, step, format, onChange, disabled }: SliderProps) {
  const id = useId()
  return (
    <div className="grid gap-1">
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="label">
          {label}
        </label>
        <output htmlFor={id} className="num text-sm text-text">
          {format(value)}
        </output>
      </div>
      <input
        id={id}
        type="range"
        className="slider"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        aria-describedby={`${id}-hint`}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <p id={`${id}-hint`} className="text-xs leading-relaxed text-muted">
        {hint}
      </p>
    </div>
  )
}
