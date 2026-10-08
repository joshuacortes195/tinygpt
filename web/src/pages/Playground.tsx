import { PlayIcon, StopIcon } from '@phosphor-icons/react'
import { useId, useMemo, useState } from 'react'
import { ModelPicker } from '../components/ModelPicker'
import { AttentionView, ProbabilityView, TextView, TokenView } from '../components/OutputViews'
import { SettingsPanel } from '../components/SettingsPanel'
import { useGeneration } from '../hooks/useGeneration'
import { useModel } from '../hooks/useModel'
import { DEFAULT_SETTINGS, type ManifestEntry, type SamplingSettings } from '../lib/types'

const EXAMPLES = ['Once upon a time', 'Lily found a little red ball', 'The dragon was sad because', 'One day, a tiny robot']

const VIEWS = [
  { id: 'text', label: 'Text' },
  { id: 'tokens', label: 'Tokens' },
  { id: 'probabilities', label: 'Probabilities' },
  { id: 'attention', label: 'Attention' },
] as const
type ViewId = (typeof VIEWS)[number]['id']

const STOP_REASON = {
  max: 'Reached the token limit.',
  end: 'The model ended the story on its own.',
  abort: 'Stopped early.',
} as const

interface Props {
  models: ManifestEntry[]
  selectedId: string
  onSelect: (id: string) => void
}

export function Playground({ models, selectedId, onSelect }: Props) {
  const promptId = useId()
  const [prompt, setPrompt] = useState(EXAMPLES[0])
  const [settings, setSettings] = useState<SamplingSettings>(DEFAULT_SETTINGS)
  const [view, setView] = useState<ViewId>('text')
  const [retry, setRetry] = useState(0)

  const entry = useMemo(() => models.find((m) => m.id === selectedId) ?? null, [models, selectedId])
  const state = useModel(entry, retry)
  const model = state.status === 'ready' ? state.model : null
  const { tokens, running, stats, error, run, stop } = useGeneration(model)

  const hasOutput = tokens.length > 0

  return (
    <div className="mx-auto grid max-w-6xl gap-8 px-4 py-8 sm:px-6 lg:grid-cols-[minmax(0,1fr)_19rem] lg:gap-12 lg:py-12">
      <div className="grid min-w-0 content-start gap-8">
        <div>
          <h1 className="max-w-[22ch] text-3xl leading-[1.1] font-semibold tracking-tight sm:text-4xl">
            A GPT I built from scratch, running in your browser.
          </h1>
          <p className="mt-3 max-w-[60ch] leading-relaxed text-muted">
            My own tokenizer, transformer and training loop, trained on children's stories. Nothing is sent to a
            server: the model downloads once and runs on your device.
          </p>
        </div>

        {/* prompt box and the button that starts the model */}
        <form
          className="grid gap-3"
          onSubmit={(e) => {
            e.preventDefault()
            if (running) stop()
            else void run(prompt, settings)
          }}
        >
          <label htmlFor={promptId} className="label">
            Prompt
          </label>
          <textarea
            id={promptId}
            className="field min-h-24 resize-y leading-relaxed"
            value={prompt}
            rows={3}
            placeholder="Start a story, or leave it empty and let the model begin"
            onChange={(e) => setPrompt(e.target.value)}
            // ctrl+enter or cmd+enter runs it without leaving the keyboard
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && model && !running) {
                e.preventDefault()
                void run(prompt, settings)
              }
            }}
          />
          <div className="flex flex-wrap items-center gap-2">
            <button type="submit" className={`btn ${running ? '' : 'btn-primary'} min-w-32`} disabled={!model}>
              {running ? <StopIcon size={16} weight="fill" /> : <PlayIcon size={16} weight="fill" />}
              {running ? 'Stop' : 'Generate'}
            </button>
            {/* one tap starting points */}
            <div className="flex flex-wrap gap-2" aria-label="Example prompts">
              {EXAMPLES.slice(1).map((example) => (
                <button
                  key={example}
                  type="button"
                  className="btn btn-ghost min-h-11 border-border px-3 text-sm font-normal text-muted"
                  onClick={() => setPrompt(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        </form>

        {/* what the model wrote, with four ways to look at it */}
        <section aria-label="Output" className="panel min-w-0">
          <div role="tablist" aria-label="Output views" className="flex gap-1 overflow-x-auto border-b border-border px-2">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                type="button"
                role="tab"
                id={`tab-${v.id}`}
                aria-selected={view === v.id}
                aria-controls="output-panel"
                onClick={() => setView(v.id)}
                className={`h-12 shrink-0 cursor-pointer border-b-2 px-3 text-[0.9375rem] transition-colors ${
                  view === v.id ? 'border-accent font-medium text-text' : 'border-transparent text-muted hover:text-text'
                }`}
              >
                {v.label}
              </button>
            ))}
          </div>

          <div id="output-panel" role="tabpanel" aria-labelledby={`tab-${view}`} className="min-h-56 p-4 sm:p-6">
            {error && (
              <p role="alert" className="mb-4 text-danger">
                Something went wrong while generating: {error}
              </p>
            )}
            {!model && state.status !== 'error' && (
              <div className="grid gap-3" aria-label="Loading the model">
                <div className="skeleton h-5 w-11/12" />
                <div className="skeleton h-5 w-4/5" />
                <div className="skeleton h-5 w-2/3" />
              </div>
            )}
            {!model && state.status === 'error' && (
              <p className="text-muted">The model did not load. Check the message next to the model picker.</p>
            )}
            {model && !hasOutput && (
              <p className="text-muted">
                Press Generate and the story shows up here one token at a time. Then use the tabs above to look
                inside the model.
              </p>
            )}
            {model && hasOutput && view === 'text' && <TextView model={model} tokens={tokens} running={running} />}
            {model && hasOutput && view === 'tokens' && <TokenView model={model} tokens={tokens} running={running} />}
            {model && hasOutput && view === 'probabilities' && (
              <ProbabilityView model={model} tokens={tokens} running={running} />
            )}
            {model && hasOutput && view === 'attention' && (
              <AttentionView model={model} tokens={tokens} running={running} />
            )}
          </div>

          {/* speed and why it stopped */}
          {stats && (
            <p className="num border-t border-border px-4 py-3 text-xs text-muted sm:px-6" aria-live="polite">
              {stats.tokens} tokens in {stats.seconds.toFixed(1)}s, {stats.tokensPerSecond.toFixed(1)} tokens/sec.{' '}
              {STOP_REASON[stats.stopped]}
            </p>
          )}
        </section>
      </div>

      {/* model choice and sampling controls */}
      <aside className="grid content-start gap-8 lg:sticky lg:top-22 lg:self-start">
        <ModelPicker
          models={models}
          selectedId={selectedId}
          onSelect={onSelect}
          state={state}
          onRetry={() => setRetry((n) => n + 1)}
          disabled={running}
        />
        <div className="border-t border-border pt-6">
          <h2 className="mb-5 text-sm font-semibold">Sampling</h2>
          <SettingsPanel settings={settings} onChange={setSettings} disabled={running} />
        </div>
      </aside>
    </div>
  )
}
