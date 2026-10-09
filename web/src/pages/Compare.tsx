import { PlayIcon, StopIcon } from '@phosphor-icons/react'
import { useId, useMemo, useState } from 'react'
import { BigModelPanel } from '../bigmodel/BigModelPanel'
import { ModelPicker } from '../components/ModelPicker'
import { TextView } from '../components/OutputViews'
import { SettingsPanel } from '../components/SettingsPanel'
import { useGeneration } from '../hooks/useGeneration'
import { useModel } from '../hooks/useModel'
import { formatCount } from '../lib/format'
import { preparePrompt, startSettings } from '../lib/recipePrompt'
import type { ExtraEntry, ManifestEntry, SamplingSettings } from '../lib/types'

interface Props {
  models: ManifestEntry[]
  defaultId: string
  extras: ExtraEntry[]
  // the two model ids to start with, when the manifest names them
  pair: string[]
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

export function Compare({ models, defaultId, extras, pair }: Props) {
  const promptId = useId()
  // start with the pair the manifest asks for, or else any two different models
  const has = (id: string | undefined) => models.some((m) => m.id === id)
  const rightId = has(pair[1]) ? pair[1] : defaultId
  const leftId = has(pair[0]) ? pair[0] : (models.find((m) => m.id !== rightId)?.id ?? rightId)
  const left = useSide(models, leftId)
  const right = useSide(models, rightId)
  // open with a prompt that suits the models being compared
  const [prompt, setPrompt] = useState(
    () => models.find((m) => m.id === rightId)?.examples?.[0] ?? 'Chocolate Chip Cookies',
  )
  const [settings, setSettings] = useState<SamplingSettings>(() => ({
    ...startSettings(models.find((m) => m.id === rightId)),
    maxTokens: 120,
  }))

  const running = left.generation.running || right.generation.running
  const ready = left.model !== null && right.model !== null

  // one after the other, so each model gets the whole machine and the speeds are fair
  async function runBoth() {
    // each side gets the prompt tidied up for its own model
    const entryFor = (id: string) => models.find((m) => m.id === id)
    const first = await left.generation.run(preparePrompt(prompt, entryFor(left.id)), settings)
    if (first?.stopped === 'abort') return
    await right.generation.run(preparePrompt(prompt, entryFor(right.id)), settings)
  }

  function stopBoth() {
    left.generation.stop()
    right.generation.stop()
  }

  return (
    <div className="frame">
      <div className="cell !px-1 sm:py-12">
        <p className="eyebrow">Compare</p>
        <h1 className="mt-3 text-[2.25rem] leading-[1.05] sm:text-[3.5rem]">Same prompt, two models.</h1>
        <p className="mt-4 max-w-[58ch] leading-relaxed text-muted">
          Both models get the same prompt, settings and seed. The only thing that changes is the model, so any
          difference in the writing comes from size and training.
        </p>
      </div>

      {/* everything below the title sits in one white card */}
      <div className="panel">
      {/* shared prompt and run button */}
      <form
        className="cell grid gap-3 border-b border-border"
        onSubmit={(e) => {
          e.preventDefault()
          if (running) stopBoth()
          else void runBoth()
        }}
      >
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <label htmlFor={promptId} className="label">
            Prompt
          </label>
          {/* sets expectations before anyone reads a recipe */}
          <p className="text-[0.8125rem] leading-snug text-muted">
            This LLM is still under development and training, so recipes may seem off or incorrect.
          </p>
        </div>
        <textarea
          id={promptId}
          className="field min-h-24 resize-y leading-relaxed"
          rows={3}
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
