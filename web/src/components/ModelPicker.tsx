import { ArrowClockwiseIcon, WarningIcon } from '@phosphor-icons/react'
import { useId } from 'react'
import type { ModelState } from '../hooks/useModel'
import { formatBytes, formatCount } from '../lib/format'
import type { ManifestEntry } from '../lib/types'

const BACKEND_LABEL = { webgpu: 'WebGPU', wasm: 'WebAssembly (CPU)', mock: 'Mock' } as const
const PRECISION_LABEL = { fp32: '32-bit weights', int8: '8-bit weights', none: '' } as const

interface Props {
  label?: string
  models: ManifestEntry[]
  selectedId: string
  onSelect: (id: string) => void
  state: ModelState
  onRetry: () => void
  disabled?: boolean
}

// dropdown to choose a model, plus what is going on with it right now
export function ModelPicker({ label = 'Model', models, selectedId, onSelect, state, onRetry, disabled }: Props) {
  const id = useId()
  const selected = models.find((m) => m.id === selectedId)

  return (
    <div className="grid gap-2">
      <label htmlFor={id} className="label">
        {label}
      </label>
      <select
        id={id}
        className="field"
        value={selectedId}
        disabled={disabled}
        onChange={(e) => onSelect(e.target.value)}
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.name}
          </option>
        ))}
      </select>
      {selected?.description && <p className="text-sm leading-relaxed text-muted">{selected.description}</p>}

      {/* download progress */}
      {state.status === 'loading' && <LoadingBar state={state} />}

      {/* which backend the model ended up on */}
      {state.status === 'ready' && state.model.generator.info && (
        <dl className="num flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
          <div>
            <dt className="sr-only">Running on</dt>
            <dd className="text-accent">{BACKEND_LABEL[state.model.generator.info.backend]}</dd>
          </div>
          {state.model.generator.info.precision !== 'none' && (
            <div>
              <dt className="sr-only">Precision</dt>
              <dd>{PRECISION_LABEL[state.model.generator.info.precision]}</dd>
            </div>
          )}
          {state.model.config.params > 0 && (
            <div>
              <dt className="sr-only">Size</dt>
              <dd>{formatCount(state.model.config.params)} parameters</dd>
            </div>
          )}
        </dl>
      )}

      {/* what went wrong and a way to try again */}
      {state.status === 'error' && (
        <div role="alert" className="grid gap-2 rounded-md border border-danger/40 p-3 text-sm">
          <p className="flex items-start gap-2 font-medium text-danger">
            <WarningIcon size={18} className="mt-0.5 shrink-0" />
            This model could not start in your browser.
          </p>
          <p className="text-muted">
            {state.message} Try the latest Chrome, Edge or Safari, or pick a smaller model.
          </p>
          <button type="button" className="btn justify-self-start" onClick={onRetry}>
            <ArrowClockwiseIcon size={16} />
            Try again
          </button>
        </div>
      )}
    </div>
  )
}

function LoadingBar({ state }: { state: Extract<ModelState, { status: 'loading' }> }) {
  const p = state.progress
  const fraction = p && p.stage === 'download' && p.total > 0 ? p.loaded / p.total : p ? 1 : 0
  const text = !p
    ? 'Getting ready'
    : p.stage === 'download'
      ? `Downloading ${formatBytes(p.loaded)} of ${formatBytes(p.total)}`
      : 'Starting the model'

  return (
    <div className="grid gap-1.5" role="status" aria-live="polite">
      <div className="h-1 overflow-hidden bg-surface-2">
        {/* grows from the left as bytes arrive */}
        <div
          className="h-full origin-left bg-accent transition-transform duration-200 ease-out"
          style={{ transform: `scaleX(${Math.max(0.02, fraction)})` }}
        />
      </div>
      <p className="num text-xs text-muted">{text}</p>
    </div>
  )
}
