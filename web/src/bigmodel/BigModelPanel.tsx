import { DownloadSimpleIcon, PlayIcon, StopIcon, WarningIcon } from '@phosphor-icons/react'
import { useEffect, useRef, useState } from 'react'
import { formatBytes } from '../lib/format'
import { modelsBaseUrl } from '../lib/manifest'
import type { ExtraEntry, SamplingSettings } from '../lib/types'
import type { BigRequest, BigResponse } from './bigmodel.worker'

type Status = 'idle' | 'loading' | 'ready' | 'running' | 'error'

interface Props {
  entry: ExtraEntry
  prompt: string
  settings: SamplingSettings
}

// the optional pretrained model: nothing downloads until the visitor asks for it
export function BigModelPanel({ entry, prompt, settings }: Props) {
  const [status, setStatus] = useState<Status>('idle')
  const [progress, setProgress] = useState<{ loaded: number; total: number } | null>(null)
  const [text, setText] = useState('')
  const [usedPrompt, setUsedPrompt] = useState('')
  const [stats, setStats] = useState<{ tokens: number; seconds: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const workerRef = useRef<Worker | null>(null)

  // shut the worker down when leaving the page
  useEffect(() => () => workerRef.current?.terminate(), [])

  function post(message: BigRequest) {
    workerRef.current?.postMessage(message)
  }

  function load() {
    const worker = new Worker(new URL('./bigmodel.worker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker
    // every update from the worker lands here
    worker.onmessage = (event: MessageEvent<BigResponse>) => {
      const msg = event.data
      if (msg.type === 'progress') setProgress({ loaded: msg.loaded, total: msg.total })
      else if (msg.type === 'loaded') setStatus('ready')
      else if (msg.type === 'text') setText((prev) => prev + msg.text)
      else if (msg.type === 'done') {
        setStats({ tokens: msg.tokens, seconds: msg.seconds })
        setStatus('ready')
      } else if (msg.type === 'error') {
        setError(msg.message)
        setStatus('error')
      }
    }
    worker.onerror = (event) => {
      setError(event.message || 'The model worker crashed.')
      setStatus('error')
    }
    setError(null)
    setStatus('loading')
    // the library wants a path on this site, not a full url
    post({ type: 'load', modelsUrl: new URL(modelsBaseUrl()).pathname, id: entry.path, dtype: entry.dtype })
  }

  function run() {
    setText('')
    setStats(null)
    setUsedPrompt(prompt)
    setStatus('running')
    post({
      type: 'run',
      prompt,
      maxTokens: settings.maxTokens,
      temperature: settings.temperature,
      topK: settings.topK,
      topP: settings.topP,
    })
  }

  const fraction = progress && progress.total > 0 ? progress.loaded / progress.total : 0

  return (
    <section className="panel grid gap-5 p-4 sm:p-5" aria-label={entry.name}>
      <div>
        <h2 className="text-lg font-semibold tracking-tight">{entry.name}</h2>
        <p className="mt-1.5 max-w-[62ch] leading-relaxed text-muted">{entry.description}</p>
      </div>

      {/* size warning and the button that starts the download */}
      {(status === 'idle' || status === 'error') && (
        <div className="grid gap-3">
          <p className="flex max-w-[62ch] items-start gap-2 text-sm leading-relaxed">
            <WarningIcon size={18} className="mt-0.5 shrink-0 text-accent" />
            <span>
              This is a {formatBytes(entry.bytes)} download and it runs on your CPU, so it is slower than the
              models above. On a phone or a metered connection, skip it.
            </span>
          </p>
          {error && (
            <p role="alert" className="text-sm text-danger">
              It did not load: {error}
            </p>
          )}
          <div>
            <button type="button" className="btn" onClick={load}>
              <DownloadSimpleIcon size={16} />
              Download {formatBytes(entry.bytes)} and load
            </button>
          </div>
        </div>
      )}

      {status === 'loading' && (
        <div className="grid max-w-md gap-1.5" role="status" aria-live="polite">
          <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
            <div
              className="h-full origin-left rounded-full bg-accent transition-transform duration-200 ease-out"
              style={{ transform: `scaleX(${Math.max(0.02, fraction)})` }}
            />
          </div>
          <p className="num text-xs text-muted">
            {progress && fraction < 1
              ? `Downloading ${formatBytes(progress.loaded)} of ${formatBytes(progress.total)}`
              : 'Starting the model'}
          </p>
        </div>
      )}

      {(status === 'ready' || status === 'running') && (
        <div className="grid gap-4">
          <div>
            {status === 'running' ? (
              <button type="button" className="btn min-w-36" onClick={() => post({ type: 'stop' })}>
                <StopIcon size={16} weight="fill" />
                Stop
              </button>
            ) : (
              <button type="button" className="btn btn-primary" onClick={run}>
                <PlayIcon size={16} weight="fill" />
                Run the prompt above
              </button>
            )}
          </div>
          <div className="min-h-24 border-t border-border pt-4">
            {text || status === 'running' ? (
              <p className="max-w-[68ch] whitespace-pre-wrap text-[1.0625rem] leading-[1.7] break-words">
                <span className="text-muted">{usedPrompt}</span>
                <span>{text}</span>
                {status === 'running' && <span className="caret" aria-hidden="true" />}
              </p>
            ) : (
              <p className="text-muted">Loaded. It uses the prompt and sampling settings from this page.</p>
            )}
          </div>
          {stats && (
            <p className="num text-xs text-muted" aria-live="polite">
              {stats.tokens} tokens in {stats.seconds.toFixed(1)}s, {(stats.tokens / stats.seconds).toFixed(1)}{' '}
              tokens/sec.
            </p>
          )}
        </div>
      )}
    </section>
  )
}
