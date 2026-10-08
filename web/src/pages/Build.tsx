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
    <div className="mx-auto grid max-w-6xl gap-14 px-4 py-8 sm:px-6 lg:py-12">
      <div>
        <h1 className="text-3xl leading-[1.1] font-semibold tracking-tight sm:text-4xl">How I built it.</h1>
        <p className="mt-3 max-w-[62ch] leading-relaxed text-muted">
          Everything that matters is written from scratch in PyTorch: the tokenizer, the transformer and the
          training loop. No pretrained weights and no model libraries. It learned English from about 2 GB of short
          children's stories.
        </p>
      </div>

      {/* pick which model the numbers below describe */}
      <section className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)] lg:gap-14">
        <div className="grid content-start gap-6">
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
            <dl className="grid grid-cols-2 gap-x-6 gap-y-5">
              <Stat label="Parameters" value={formatCount(config.params)} />
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
              />
              <Stat
                label="Download size"
                value={`${formatBytes(config.files.int8?.bytes)} to ${formatBytes(config.files.fp32?.bytes)}`}
              />
            </dl>
          )}
        </div>

        <div className="min-w-0">
          <h2 className="mb-4 text-lg font-semibold tracking-tight">Loss during training</h2>
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

      {/* the architecture, top to bottom */}
      {config && (
        <section className="grid gap-8 border-t border-border pt-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)] lg:gap-14">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">What happens to your text</h2>
            <p className="mt-3 max-w-[52ch] leading-relaxed text-muted">
              The model only ever does one thing: given some tokens, guess the next one. To write a story it
              guesses, adds the guess to the text, and goes again. The highlighted step is the one the Attention
              tab in the playground lets you look inside.
            </p>
          </div>
          <ArchitectureDiagram config={config} />
        </section>
      )}

      {/* write-ups */}
      <section className="border-t border-border pt-12">
        <h2 className="text-lg font-semibold tracking-tight">Notes from building it</h2>
        <div className="mt-6 grid gap-x-12 gap-y-8 sm:grid-cols-2">
          {NOTES.map((note) => (
            <article key={note.title} className="max-w-[48ch]">
              <h3 className="font-medium">{note.title}</h3>
              <p className="mt-1.5 leading-relaxed text-muted">{note.body}</p>
              <a
                href={learningLink(note.anchor)}
                className="mt-2 inline-flex min-h-11 items-center gap-1 text-sm font-medium text-accent hover:underline"
              >
                Read the notes
                <ArrowUpRightIcon size={14} />
              </a>
            </article>
          ))}
          <article className="max-w-[48ch]">
            <h3 className="font-medium">All of the code</h3>
            <p className="mt-1.5 leading-relaxed text-muted">
              The model, the training scripts, the tests and this website are in one repository.
            </p>
            <a
              href={REPO_URL}
              className="mt-2 inline-flex min-h-11 items-center gap-1 text-sm font-medium text-accent hover:underline"
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

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-sm text-muted">{label}</dt>
      <dd className="num mt-0.5 text-lg">{value}</dd>
    </div>
  )
}
