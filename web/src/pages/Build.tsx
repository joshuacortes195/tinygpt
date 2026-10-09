import { ArrowUpRightIcon } from '@phosphor-icons/react'
import { useEffect, useId, useMemo, useState } from 'react'
import { ArchitectureDiagram } from '../components/ArchitectureDiagram'
import { LossChart } from '../components/LossChart'
import { formatBytes, formatCount, formatDuration } from '../lib/format'
import { learningLink, REPO_URL } from '../lib/links'
import { loadConfig, loadMetrics } from '../lib/manifest'
import { MOCK_ENTRY } from '../lib/models'
import type { ManifestEntry, Metrics, ModelConfig } from '../lib/types'

// short write-ups, each one links to the longer notes in the repo
const NOTES = [
  {
    title: 'A tokenizer with no unknown words',
    body: 'Byte-level BPE starts from the 256 possible bytes and learns merges for common pairs. Any text in any language can be encoded, and the browser version has to match the Python one id for id.',
    anchor: 'phase-1-byte-level-bpe-tokenizer',
  },
  {
    title: 'Attention written out by hand',
    body: 'No library attention layers. Queries, keys, values, the causal mask and the softmax are all spelled out, which is also why the site can show you the attention weights.',
    anchor: 'phase-2-the-gpt-model-and-the-training-loop',
  },
  {
    title: 'A training run you can interrupt',
    body: 'Checkpoints store the model, the optimizer and every random number generator. A test stops a run half way, resumes it, and checks the weights match a run that never stopped.',
    anchor: 'phase-2-the-gpt-model-and-the-training-loop',
  },
  {
    title: 'From PyTorch to your browser',
    body: 'The trained model is exported to ONNX and checked number by number against PyTorch. An 8-bit copy is about a quarter of the size for devices without WebGPU.',
    anchor: 'phase-3-onnx-export-and-parity',
  },
  {
    title: 'A site that does not care which model it runs',
    body: 'Models are listed in one manifest file. Adding a new one means dropping in a folder and adding a line, with no code changes. The same interface runs a mock model in tests.',
    anchor: 'phase-4-the-web-app',
  },
]

interface Props {
  models: ManifestEntry[]
  defaultId: string
}

export function Build({ models, defaultId }: Props) {
  const selectId = useId()
  const real = useMemo(() => models.filter((m) => m.id !== MOCK_ENTRY.id), [models])
  const [id, setId] = useState(real.some((m) => m.id === defaultId) ? defaultId : (real[0]?.id ?? ''))
  const [data, setData] = useState<{ config: ModelConfig; metrics: Metrics | null } | null>(null)
  const [error, setError] = useState<string | null>(null)

  // stats and loss curve come straight from the model's own files
  useEffect(() => {
    const entry = real.find((m) => m.id === id)
    if (!entry) return
    let cancelled = false
    setData(null)
    setError(null)
    Promise.all([loadConfig(entry), loadMetrics(entry).catch(() => null)])
      .then(([config, metrics]) => {
        if (!cancelled) setData({ config, metrics })
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      })
    return () => {
      cancelled = true
    }
  }, [id, real])

  const config = data?.config

  return (
    <div className="frame">
      <div className="cell !px-1 sm:py-12">
        <p className="eyebrow">How I built it</p>
        <h1 className="mt-3 max-w-[22ch] text-[2.25rem] leading-[1.05] sm:text-[3.5rem]">
          Every part written by hand.
        </h1>
        <p className="mt-4 max-w-[60ch] leading-relaxed text-muted">
          The tokenizer, the transformer and the training loop are all my own PyTorch code. No pretrained weights
          and no model libraries. Each model here learned to write from nothing but its training text.
        </p>
      </div>

      {/* pick which model the numbers below describe */}
      <section className="panel grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="cell grid content-start gap-6">
          <div className="grid max-w-xs gap-2">
            <label htmlFor={selectId} className="label">
              Show details for
            </label>
            <select id={selectId} className="field" value={id} onChange={(e) => setId(e.target.value)}>
              {real.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          {error && (
            <p role="alert" className="text-danger">
              Could not load this model's details: {error}
            </p>
          )}
          {!config && !error && <div className="skeleton h-64 w-full" aria-label="Loading model details" />}

          {/* model facts read from config.json */}
          {config && (
            <dl className="grid grid-cols-2 overflow-hidden rounded-lg border-t border-l border-border">
              <Stat label="Parameters" value={formatCount(config.params)} strong />
              <Stat label="Layers" value={String(config.n_layer)} />
              <Stat label="Attention heads" value={`${config.n_head} per layer`} />
              <Stat label="Vector width" value={String(config.n_embd)} />
              <Stat label="Context window" value={`${config.block_size} tokens`} />
              <Stat label="Vocabulary" value={`${config.vocab_size.toLocaleString()} tokens`} />
              <Stat label="Tokens trained on" value={formatCount(config.training_tokens)} />
              <Stat label="Training time" value={formatDuration(config.training_seconds)} />
              <Stat
                label="Validation loss"
                value={config.final_val_loss === null ? 'n/a' : config.final_val_loss.toFixed(3)}
                strong
              />
              <Stat
                label="Download size"
                value={`${formatBytes(config.files.int8?.bytes)} to ${formatBytes(config.files.fp32?.bytes)}`}
              />
            </dl>
          )}
        </div>

        <div className="cell min-w-0 border-t border-border lg:border-t-0 lg:border-l">
          <h2 className="eyebrow mb-5">Loss during training</h2>
          {data ? (
            data.metrics ? (
              <LossChart metrics={data.metrics} />
            ) : (
              <p className="text-muted">No training history was saved for this model.</p>
            )
          ) : (
            !error && <div className="skeleton aspect-[2/1] w-full" aria-label="Loading the loss curve" />
          )}
        </div>
      </section>

      {/* the architecture, left to right */}
      {config && (
        <section className="cell panel mt-5">
          <h2 className="eyebrow">What happens to your text</h2>
          <p className="mt-4 mb-8 max-w-[64ch] leading-relaxed text-muted">
            The model only ever does one thing: given some tokens, guess the next one. To write, it guesses, adds
            the guess to the text, and goes again. The highlighted step is the one the Attention tab on the Try it
            page lets you look inside.
          </p>
          <ArchitectureDiagram config={config} />
        </section>
      )}

      {/* write-ups */}
      <section className="panel mt-5">
        <h2 className="eyebrow cell !pb-0">Notes from building it</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3">
          {NOTES.map((note, i) => (
            <article key={note.title} className="cell flex flex-col">
              <p className="num text-xs text-accent">{String(i + 1).padStart(2, '0')}</p>
              <h3 className="mt-2 font-medium">{note.title}</h3>
              <p className="mt-2 max-w-[46ch] flex-1 leading-relaxed text-muted">{note.body}</p>
              <a
                href={learningLink(note.anchor)}
                className="mt-2 inline-flex min-h-11 items-center gap-1 self-start text-sm font-medium text-accent hover:underline"
              >
                Read the notes
                <ArrowUpRightIcon size={14} />
              </a>
            </article>
          ))}
          <article className="cell flex flex-col">
            <p className="num text-xs text-accent">{String(NOTES.length + 1).padStart(2, '0')}</p>
            <h3 className="mt-2 font-medium">All of the code</h3>
            <p className="mt-2 max-w-[46ch] flex-1 leading-relaxed text-muted">
              The model, the training scripts, the tests and this website are in one repository.
            </p>
            <a
              href={REPO_URL}
              className="mt-2 inline-flex min-h-11 items-center gap-1 self-start text-sm font-medium text-accent hover:underline"
            >
              View the source
              <ArrowUpRightIcon size={14} />
            </a>
          </article>
        </div>
      </section>
    </div>
  )
}

// one box in the spec sheet
function Stat({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="border-r border-b border-border p-3 sm:p-4">
      <dt className="label">{label}</dt>
      <dd className={`num mt-1.5 text-lg ${strong ? 'text-accent' : ''}`}>{value}</dd>
    </div>
  )
}
