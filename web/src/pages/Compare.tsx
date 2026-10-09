import { PlayIcon, StopIcon } from '@phosphor-icons/react'
import { useId, useMemo, useState } from 'react'
import { BigModelPanel } from '../bigmodel/BigModelPanel'
import { ModelPicker } from '../components/ModelPicker'
import { TextView } from '../components/OutputViews'
import { SettingsPanel } from '../components/SettingsPanel'
import { useGeneration } from '../hooks/useGeneration'
import { useModel } from '../hooks/useModel'
import { formatCount } from '../lib/format'
import { DEFAULT_SETTINGS, type ExtraEntry, type ManifestEntry, type SamplingSettings } from '../lib/types'

interface Props {
  models: ManifestEntry[]
  defaultId: string
  extras: ExtraEntry[]
}

// one side of the comparison: its own model, its own output
function useSide(models: ManifestEntry[], initialId: string) {
  const [id, setId] = useState(initialId)
  const [retry, setRetry] = useState(0)
  const entry = useMemo(() => models.find((m) => m.id === id) ?? null, [models, id])
  const state = useModel(entry, retry)
  const model = state.status === 'ready' ? state.model : null
  const generation = useGeneration(model)
  return { id, setId, state, model, generation, retry: () => setRetry((n) => n + 1) }
}

export function Compare({ models, defaultId, extras }: Props) {
  const promptId = useId()
  // start with two different models when there are at least two
  const otherId = models.find((m) => m.id !== defaultId)?.id ?? defaultId
  const left = useSide(models, otherId)
  const right = useSide(models, defaultId)
  const [prompt, setPrompt] = useState('Once upon a time, there was a little dog named')
  const [settings, setSettings] = useState<SamplingSettings>({ ...DEFAULT_SETTINGS, maxTokens: 80 })

  const running = left.generation.running || right.generation.running
  const ready = left.model !== null && right.model !== null

  // one after the other, so each model gets the whole machine and the speeds are fair
  async function runBoth() {
    const first = await left.generation.run(prompt, settings)
    if (first?.stopped === 'abort') return
    await right.generation.run(prompt, settings)
  }

  function stopBoth() {
    left.generation.stop()
    right.generation.stop()
  }

  return (
    <div className="frame">
      <div className="cell border-b border-border sm:py-12">
        <p className="eyebrow">Compare</p>
        <h1 className="mt-4 text-[2rem] leading-[1.05] font-semibold tracking-tight sm:text-5xl">Same prompt, two models.</h1>
        <p className="mt-4 max-w-[58ch] leading-relaxed text-muted">
          Both models get the same prompt, settings and seed. The only thing that changes is the model, so any
          difference in the writing comes from size and training.
        </p>
      </div>

      {/* shared prompt and run button */}
      <form
        className="cell grid gap-3 border-b border-border"
        onSubmit={(e) => {
          e.preventDefault()
          if (running) stopBoth()
          else void runBoth()
        }}
      >
        <label htmlFor={promptId} className="label">
          Prompt
        </label>
        <textarea
          id={promptId}
          className="field min-h-20 resize-y leading-relaxed"
          rows={2}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <div>
          <button type="submit" className={`btn ${running ? '' : 'btn-primary'} min-w-36`} disabled={!ready}>
            {running ? <StopIcon size={16} weight="fill" /> : <PlayIcon size={16} weight="fill" />}
            {running ? 'Stop' : 'Run both'}
          </button>
        </div>
      </form>

      {/* the two outputs, side by side on wide screens and stacked on phones */}
      <div className="grid md:grid-cols-2">
        <Side title="Model A" models={models} side={left} disabled={running} />
        <Side title="Model B" models={models} side={right} disabled={running} divided />
      </div>

      {/* optional bigger models, each behind its own download button */}
      {extras.map((extra) => (
        <BigModelPanel key={extra.id} entry={extra} prompt={prompt} settings={settings} />
      ))}

      <details className="cell border-t border-border">
        <summary className="eyebrow cursor-pointer py-2">Sampling settings</summary>
        <div className="mt-4 max-w-sm">
          <SettingsPanel settings={settings} onChange={setSettings} disabled={running} />
        </div>
      </details>
    </div>
  )
}

function Side({
  title,
  models,
  side,
  disabled,
  divided,
}: {
  title: string
  models: ManifestEntry[]
  side: ReturnType<typeof useSide>
  disabled: boolean
  divided?: boolean
}) {
  const { model, generation } = side
  return (
    <section
      // a line between the two sides: above when stacked, to the left when side by side
      className={`cell grid min-w-0 content-start gap-5 ${divided ? 'border-t border-border md:border-t-0 md:border-l' : ''}`}
      aria-label={title}
    >
      <ModelPicker
        label={title}
        models={models}
        selectedId={side.id}
        onSelect={side.setId}
        state={side.state}
        onRetry={side.retry}
        disabled={disabled}
      />

      <div className="min-h-40 border-t border-border pt-5">
        {generation.error && (
          <p role="alert" className="text-danger">
            {generation.error}
          </p>
        )}
        {model && generation.tokens.length > 0 ? (
          <TextView model={model} tokens={generation.tokens} running={generation.running} />
        ) : (
          <p className="text-muted">{model ? 'Output shows up here.' : 'Waiting for the model to load.'}</p>
        )}
      </div>

      {/* speed and size, so the tradeoff is visible */}
      {model && generation.stats && (
        <dl className="num grid grid-cols-3 gap-3 border-t border-border pt-4 text-[0.8125rem]">
          <div>
            <dt className="text-xs text-muted">Speed</dt>
            <dd className="text-accent">{generation.stats.tokensPerSecond.toFixed(1)} tok/s</dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Tokens</dt>
            <dd>{generation.stats.tokens}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Parameters</dt>
            <dd>{model.config.params ? formatCount(model.config.params) : 'n/a'}</dd>
          </div>
        </dl>
      )}
    </section>
  )
}
