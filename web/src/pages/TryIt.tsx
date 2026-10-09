import { PlayIcon, StopIcon } from '@phosphor-icons/react'
import { useId, useMemo, useState } from 'react'
import { ModelPicker } from '../components/ModelPicker'
import { AttentionView, ProbabilityView, TextView, TokenView } from '../components/OutputViews'
import { SettingsPanel } from '../components/SettingsPanel'
import { useGeneration } from '../hooks/useGeneration'
import { useModel } from '../hooks/useModel'
import { preparePrompt, startSettings } from '../lib/recipePrompt'
import type { ManifestEntry, SamplingSettings } from '../lib/types'

// starting prompts for models that don't bring their own
const EXAMPLES = ['Chocolate Chip Cookies', 'Chicken Noodle Soup', 'Banana Bread', 'Garlic Butter Pasta']

const VIEWS = [
  { id: 'text', label: 'Text' },
  { id: 'tokens', label: 'Tokens' },
  { id: 'probabilities', label: 'Probabilities' },
  { id: 'attention', label: 'Attention' },
] as const
type ViewId = (typeof VIEWS)[number]['id']

const STOP_REASON = {
  max: 'Reached the token limit.',
  end: 'The model decided it was finished.',
  abort: 'Stopped early.',
} as const

// the example prompts a model comes with
function examplesFor(entry: ManifestEntry | null | undefined): string[] {
  return entry?.examples?.length ? entry.examples : EXAMPLES
}

interface Props {
  models: ManifestEntry[]
  selectedId: string
  onSelect: (id: string) => void
}

export function TryIt({ models, selectedId, onSelect }: Props) {
  const promptId = useId()
  const entry = useMemo(() => models.find((m) => m.id === selectedId) ?? null, [models, selectedId])
  const examples = examplesFor(entry)
  const [prompt, setPrompt] = useState(() => examples[0])
  // true once the visitor has typed a prompt of their own
  const [typed, setTyped] = useState(false)
  const [settings, setSettings] = useState<SamplingSettings>(() => startSettings(entry))
  const [view, setView] = useState<ViewId>('text')
  const [retry, setRetry] = useState(0)

  const state = useModel(entry, retry)
  const model = state.status === 'ready' ? state.model : null
  const { tokens, running, stats, error, run, stop } = useGeneration(model)

  const hasOutput = tokens.length > 0
  // recipe models take the name of a dish instead of free text
  const isRecipe = entry?.format === 'recipe'

  // switching models swaps in that model's first example, unless you wrote your own prompt
  function pickModel(id: string) {
    onSelect(id)
    if (!typed) setPrompt(examplesFor(models.find((m) => m.id === id))[0])
  }

  return (
    <div className="frame">
      {/* intro */}
      <div className="cell !px-1 sm:py-12">
        <p className="eyebrow">Runs on your device</p>
        <h1 className="mt-3 max-w-[20ch] text-[2.25rem] leading-[1.05] sm:text-[3.5rem]">
          A GPT I built from scratch, and taught to write recipes.
        </h1>
        <p className="mt-4 max-w-[58ch] leading-relaxed text-muted">
          My own tokenizer, transformer and training loop. Nothing is sent to a server: the model downloads once
          and runs on your device. Give it the name of a dish and watch it write, then look inside.
        </p>
      </div>

      <div className="panel grid lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-w-0 lg:border-r lg:border-border">
          {/* prompt box and the button that starts the model */}
          <form
            className="cell grid gap-3 border-b border-border"
            onSubmit={(e) => {
              e.preventDefault()
              if (running) stop()
              else void run(preparePrompt(prompt, entry), settings)
            }}
          >
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <label htmlFor={promptId} className="label">
                {isRecipe ? 'Name a dish' : 'Prompt'}
              </label>
              {/* sets expectations before anyone reads a recipe */}
              <p className="text-[0.8125rem] leading-snug text-muted">
                This LLM is still under development and training, so recipes may seem off or incorrect.
              </p>
            </div>
            <textarea
              id={promptId}
              className="field min-h-24 resize-y leading-relaxed"
              value={prompt}
              rows={3}
              placeholder={isRecipe ? 'Pizza, ramen noodles, banana bread...' : 'Start it off, or leave this empty and let the model begin'}
              onChange={(e) => {
                setPrompt(e.target.value)
                setTyped(true)
              }}
              // ctrl+enter or cmd+enter runs it without leaving the keyboard
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && model && !running) {
                  e.preventDefault()
                  void run(preparePrompt(prompt, entry), settings)
                }
              }}
            />
            <div className="flex flex-wrap items-center gap-2">
              <button type="submit" className={`btn ${running ? '' : 'btn-primary'} min-w-32`} disabled={!model}>
                {running ? <StopIcon size={16} weight="fill" /> : <PlayIcon size={16} weight="fill" />}
                {running ? 'Stop' : 'Generate'}
              </button>
              {/* one tap starting points, labelled by their first line */}
              <div className="flex flex-wrap gap-2" aria-label="Example prompts">
                {examples.map((example) => (
                  <button
                    key={example}
                    type="button"
                    className="chip"
                    onClick={() => {
                      setPrompt(example)
                      setTyped(false)
                    }}
                  >
                    {example.split('\n')[0]}
                  </button>
                ))}
              </div>
            </div>
          </form>

          {/* what the model wrote, with four ways to look at it */}
          <section aria-label="Output" className="min-w-0">
            <div role="tablist" aria-label="Output views" className="flex overflow-x-auto border-b border-border px-2 sm:px-5">
              {VIEWS.map((v) => (
                <button
                  key={v.id}
                  type="button"
                  role="tab"
                  id={`tab-${v.id}`}
                  aria-selected={view === v.id}
                  aria-controls="output-panel"
                  onClick={() => setView(v.id)}
                  className={`h-12 shrink-0 cursor-pointer border-b-2 px-3 text-[0.9375rem] font-medium transition-colors ${
                    view === v.id ? 'border-accent text-text' : 'border-transparent text-muted hover:text-text'
                  }`}
                >
                  {v.label}
                </button>
              ))}
            </div>

            <div id="output-panel" role="tabpanel" aria-labelledby={`tab-${view}`} className="cell min-h-72">
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
                <p className="max-w-[52ch] leading-relaxed text-muted">
                  Press Generate and the text shows up here one token at a time. Then use the tabs above to look
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
              <p className="num border-t border-border px-4 py-3 text-xs text-muted sm:px-7" aria-live="polite">
                {stats.tokens} tokens in {stats.seconds.toFixed(1)}s, {stats.tokensPerSecond.toFixed(1)} tokens/sec.{' '}
                {STOP_REASON[stats.stopped]}
              </p>
            )}
          </section>
        </div>

        {/* model choice and sampling controls */}
        <aside className="cell grid content-start gap-8 border-t border-border bg-surface-2/50 lg:border-t-0">
          <ModelPicker
            models={models}
            selectedId={selectedId}
            onSelect={pickModel}
            state={state}
            onRetry={() => setRetry((n) => n + 1)}
            disabled={running}
          />
          <div className="border-t border-border pt-6">
            <h2 className="eyebrow mb-5">Sampling</h2>
            <SettingsPanel settings={settings} onChange={setSettings} disabled={running} />
          </div>
        </aside>
      </div>
    </div>
  )
}
