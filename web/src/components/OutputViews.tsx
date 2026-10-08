import { useEffect, useMemo, useRef, useState } from 'react'
import { formatPercent, visibleToken } from '../lib/format'
import type { LoadedModel } from '../lib/models'
import type { TokenInfo } from '../lib/types'

interface ViewProps {
  model: LoadedModel
  tokens: TokenInfo[]
  running: boolean
}

// a steady color for each token id, so the same token always looks the same
function tokenHue(id: number): number {
  return (id * 137.5) % 360
}

// plain reading view: prompt in gray, the model's writing in normal text
export function TextView({ model, tokens, running }: ViewProps) {
  const promptText = useMemo(
    () => model.tokenizer.decode(tokens.filter((t) => t.fromPrompt).map((t) => t.id)),
    [model, tokens],
  )
  const outputText = useMemo(
    () => model.tokenizer.decode(tokens.filter((t) => !t.fromPrompt).map((t) => t.id)),
    [model, tokens],
  )
  return (
    <p className="max-w-[68ch] whitespace-pre-wrap text-[1.0625rem] leading-[1.7] break-words">
      <span className="text-muted">{promptText}</span>
      <span>{outputText}</span>
      {running && <span className="caret" aria-hidden="true" />}
    </p>
  )
}

// every token as a colored chip with its id, so you can see how text gets split up
export function TokenView({ model, tokens }: ViewProps) {
  return (
    <div>
      <ul className="flex flex-wrap gap-x-1 gap-y-2" aria-label="Tokens">
        {tokens.map((token, i) => (
          <li
            key={i}
            className={`grid justify-items-center gap-0.5 ${token.fromPrompt ? '' : 'tok-new'}`}
            title={`Token id ${token.id}`}
          >
            <span
              className="tok num px-1 py-0.5 text-[0.9375rem] leading-snug whitespace-pre"
              style={{ '--h': tokenHue(token.id) } as React.CSSProperties}
            >
              {visibleToken(model.tokenizer.tokenLabel(token.id))}
            </span>
            <span className="num text-[0.6875rem] leading-none text-muted">{token.id}</span>
          </li>
        ))}
      </ul>
      <p className="mt-5 text-sm leading-relaxed text-muted">
        {tokens.length} tokens. A dot is a space and ↵ is a new line. Common words are one token, rare words get
        built from pieces.
      </p>
    </div>
  )
}

// tap a generated token to see what else the model was considering at that step
export function ProbabilityView({ model, tokens }: ViewProps) {
  const firstGenerated = tokens.findIndex((t) => !t.fromPrompt)
  const [selected, setSelected] = useState<number | null>(null)
  // fall back to the first generated token when nothing valid is picked
  const active = selected !== null && tokens[selected] && !tokens[selected].fromPrompt ? selected : firstGenerated
  const token = active >= 0 ? tokens[active] : null

  if (firstGenerated < 0) {
    return <p className="text-muted">Generate some text first, then come back to see the model's other guesses.</p>
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_17rem] lg:items-start">
      <div>
        <p className="text-[1.0625rem] leading-[2] break-words whitespace-pre-wrap">
          {tokens.map((t, i) =>
            t.fromPrompt ? (
              <span key={i} className="text-muted">
                {model.tokenizer.tokenLabel(t.id)}
              </span>
            ) : (
              <button
                key={i}
                type="button"
                onClick={() => setSelected(i)}
                onMouseEnter={() => setSelected(i)}
                aria-pressed={i === active}
                aria-label={`${visibleToken(model.tokenizer.tokenLabel(t.id))}, ${formatPercent(t.prob ?? 0)}`}
                className={`cursor-pointer rounded-[3px] py-0.5 whitespace-pre-wrap ${
                  i === active ? 'outline-2 outline-offset-1 outline-accent' : ''
                }`}
                style={{
                  // the less sure the model was, the stronger the highlight
                  backgroundColor: `color-mix(in oklch, var(--accent) ${Math.round((1 - (t.prob ?? 0)) * 42)}%, transparent)`,
                }}
              >
                {model.tokenizer.tokenLabel(t.id)}
              </button>
            ),
          )}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-muted">
          Stronger color means the model was less sure about that token. Tap one to see its other options.
        </p>
      </div>

      {/* the top 10 guesses for the picked step */}
      {token && (
        <div key={active} className="reveal" aria-live="polite">
          <h3 className="text-sm font-medium">
            Top guesses for token {active - firstGenerated + 1}
          </h3>
          <ol className="mt-3 grid gap-1.5">
            {(token.top ?? []).map((candidate) => {
              const chosen = candidate.id === token.id
              return (
                <li key={candidate.id} className="grid grid-cols-[6.5rem_minmax(0,1fr)_3.5rem] items-center gap-2 text-sm">
                  <span className={`num truncate ${chosen ? 'font-semibold text-text' : 'text-muted'}`}>
                    {visibleToken(model.tokenizer.tokenLabel(candidate.id))}
                  </span>
                  {/* bar length is the probability */}
                  <span
                    className={`h-2 origin-left rounded-full ${chosen ? 'bg-accent' : 'bg-border-strong'}`}
                    style={{ width: `${Math.max(1.5, candidate.prob * 100)}%` }}
                  />
                  <span className="num text-right text-xs text-muted">{formatPercent(candidate.prob)}</span>
                </li>
              )
            })}
          </ol>
          {/* the picked token can fall outside the top 10 when sampling gets lucky */}
          {!token.top?.some((c) => c.id === token.id) && (
            <p className="mt-3 text-sm text-muted">
              The model picked <span className="num text-text">{visibleToken(model.tokenizer.tokenLabel(token.id))}</span>{' '}
              at {formatPercent(token.prob ?? 0)}, outside its top 10.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

// shows which earlier tokens a token looked at, for one layer and head
export function AttentionView({ model, tokens, running }: ViewProps) {
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [focus, setFocus] = useState<number | null>(null)
  const [state, setState] = useState<
    | { status: 'idle' | 'loading' }
    | { status: 'ready'; size: number; layers: number; heads: number; data: Float32Array; ids: number[] }
    | { status: 'error'; message: string }
  >({ status: 'idle' })

  // the model only sees the last block_size tokens, so that is what gets drawn
  const ids = useMemo(() => tokens.map((t) => t.id).slice(-model.config.block_size), [tokens, model])
  const getAttention = model.generator.attention

  // ask the model for its attention maps once the text has stopped changing
  useEffect(() => {
    if (!getAttention || running || ids.length < 2) {
      setState({ status: 'idle' })
      return
    }
    let cancelled = false
    setState({ status: 'loading' })
    getAttention(ids)
      .then((a) => {
        if (!cancelled) setState({ status: 'ready', ...a, ids })
      })
      .catch((err: unknown) => {
        if (!cancelled) setState({ status: 'error', message: err instanceof Error ? err.message : String(err) })
      })
    return () => {
      cancelled = true
    }
  }, [getAttention, ids, running])

  if (!getAttention) {
    return <p className="text-muted">This model was exported without attention maps, so there is nothing to show here.</p>
  }
  if (running) return <p className="text-muted">Attention shows up once the model finishes writing.</p>
  if (ids.length < 2) return <p className="text-muted">Generate some text first to see what the model pays attention to.</p>
  if (state.status === 'error') return <p className="text-danger">Could not read attention: {state.message}</p>
  if (state.status !== 'ready') return <div className="skeleton h-40 w-full" aria-label="Loading attention" />

  const { size, layers, heads, data } = state
  const l = Math.min(layer, layers - 1)
  const h = Math.min(head, heads - 1)
  const row = focus !== null && focus < size ? focus : size - 1
  // where this layer and head start inside the flat array
  const offset = (l * heads + h) * size * size
  const weights = data.subarray(offset + row * size, offset + row * size + size)
  const strongest = weights.reduce((best, w) => Math.max(best, w), 0) || 1

  return (
    <div className="grid gap-5">
      <div className="flex flex-wrap gap-x-6 gap-y-3">
        <Stepper label="Layer" count={layers} value={l} onChange={setLayer} />
        <Stepper label="Head" count={heads} value={h} onChange={setHead} />
      </div>

      {/* tap a token, the tokens before it light up by how much it looked at them */}
      <div>
        <p className="text-[1.0625rem] leading-[2.1] break-words whitespace-pre-wrap">
          {state.ids.map((id, i) => {
            const weight = i <= row ? weights[i] / strongest : 0
            return (
              <button
                key={i}
                type="button"
                onClick={() => setFocus(i)}
                aria-pressed={i === row}
                className={`cursor-pointer rounded-[3px] py-0.5 whitespace-pre-wrap ${
                  i === row ? 'outline-2 outline-offset-1 outline-text' : ''
                } ${i > row ? 'text-muted' : ''}`}
                style={{ backgroundColor: `oklch(var(--heat) / ${(weight * 0.85).toFixed(3)})` }}
              >
                {model.tokenizer.tokenLabel(id)}
              </button>
            )
          })}
        </p>
        <p className="mt-3 text-sm leading-relaxed text-muted">
          The outlined token is the one doing the looking. Brighter tokens got more of its attention. Tokens after
          it are grayed out because the model is not allowed to look ahead.
        </p>
      </div>

      <AttentionMatrix data={data} offset={offset} size={size} row={row} onPickRow={setFocus} />
    </div>
  )
}

// row of numbered buttons for choosing a layer or a head
function Stepper({ label, count, value, onChange }: { label: string; count: number; value: number; onChange: (v: number) => void }) {
  return (
    <div className="grid gap-1.5">
      <span className="label">{label}</span>
      <div className="flex flex-wrap gap-1" role="group" aria-label={label}>
        {Array.from({ length: count }, (_, i) => (
          <button
            key={i}
            type="button"
            aria-pressed={i === value}
            onClick={() => onChange(i)}
            className={`num h-11 min-w-11 cursor-pointer rounded-lg border px-2 text-sm transition-colors sm:h-9 sm:min-w-9 ${
              i === value
                ? 'border-transparent bg-accent text-accent-fg'
                : 'border-border-strong bg-surface text-text hover:border-text'
            }`}
          >
            {i + 1}
          </button>
        ))}
      </div>
    </div>
  )
}

// the full grid: each row is a token, each column is a token it could look at
function AttentionMatrix({
  data,
  offset,
  size,
  row,
  onPickRow,
}: {
  data: Float32Array
  offset: number
  size: number
  row: number
  onPickRow: (row: number) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // paints one pixel per pair of tokens
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    canvas.width = size
    canvas.height = size
    // read the theme colors so the grid matches light and dark mode
    const styles = getComputedStyle(document.documentElement)
    const probe = document.createElement('canvas').getContext('2d')!
    const toRgb = (color: string) => {
      probe.fillStyle = color
      probe.fillRect(0, 0, 1, 1)
      return probe.getImageData(0, 0, 1, 1).data
    }
    const low = toRgb(styles.getPropertyValue('--surface-2'))
    const high = toRgb(`oklch(${styles.getPropertyValue('--heat')})`)

    const image = ctx.createImageData(size, size)
    for (let r = 0; r < size; r++) {
      // scale each row by its own strongest weight so faint rows are still readable
      let max = 0
      for (let c = 0; c <= r; c++) max = Math.max(max, data[offset + r * size + c])
      for (let c = 0; c < size; c++) {
        const w = c <= r && max > 0 ? Math.sqrt(data[offset + r * size + c] / max) : 0
        const p = (r * size + c) * 4
        image.data[p] = low[0] + (high[0] - low[0]) * w
        image.data[p + 1] = low[1] + (high[1] - low[1]) * w
        image.data[p + 2] = low[2] + (high[2] - low[2]) * w
        // the top right half is the future, leave it empty
        image.data[p + 3] = c <= r ? 255 : 0
      }
    }
    ctx.putImageData(image, 0, 0)
  }, [data, offset, size])

  return (
    <details className="group">
      <summary className="cursor-pointer py-2 text-sm font-medium text-text">Full attention grid</summary>
      <div className="mt-2 grid gap-2">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label="Attention grid. Each row is a token and each column is an earlier token it looked at."
          className="aspect-square w-full max-w-md cursor-crosshair rounded-lg border border-border [image-rendering:pixelated]"
          // clicking a row picks that token above
          onClick={(e) => {
            const box = e.currentTarget.getBoundingClientRect()
            onPickRow(Math.min(size - 1, Math.floor(((e.clientY - box.top) / box.height) * size)))
          }}
        />
        <p className="max-w-md text-sm leading-relaxed text-muted">
          Rows are tokens, top to bottom. Columns are the tokens they can look at. Row {row + 1} is the one picked
          above. The empty corner is the future, which the causal mask hides.
        </p>
      </div>
    </details>
  )
}
