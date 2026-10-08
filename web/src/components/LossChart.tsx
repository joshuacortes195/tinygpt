import { useMemo, useState } from 'react'
import type { Metrics } from '../lib/types'

const W = 640
const H = 300
const PAD = { top: 16, right: 16, bottom: 36, left: 44 }

// round numbers for the axis labels
function ticks(min: number, max: number, count: number): number[] {
  const step = (max - min) / count
  return Array.from({ length: count + 1 }, (_, i) => min + step * i)
}

// training and validation loss over time, drawn as an svg line chart
export function LossChart({ metrics }: { metrics: Metrics }) {
  const [hover, setHover] = useState<number | null>(null)

  const chart = useMemo(() => {
    const train = metrics.train
    const val = metrics.val
    if (train.length < 2) return null
    const maxStep = Math.max(train[train.length - 1].step, val.length ? val[val.length - 1].step : 0)
    // skip the very first points when picking the top of the axis, they are huge and flatten everything else
    const settled = train.slice(Math.min(3, train.length - 1)).map((p) => p.loss)
    const all = [...settled, ...val.map((p) => p.loss)]
    const maxLoss = Math.max(...all)
    const minLoss = Math.min(...all)
    const top = maxLoss + (maxLoss - minLoss) * 0.08
    const bottom = Math.max(0, minLoss - (maxLoss - minLoss) * 0.08)

    const x = (step: number) => PAD.left + (step / maxStep) * (W - PAD.left - PAD.right)
    const y = (loss: number) =>
      PAD.top + (1 - (Math.min(loss, top) - bottom) / (top - bottom)) * (H - PAD.top - PAD.bottom)
    const line = (points: { step: number; loss: number }[]) =>
      points.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`).join(' ')

    return { train, val, maxStep, top, bottom, x, y, trainPath: line(train), valPath: line(val) }
  }, [metrics])

  if (!chart) return <p className="text-muted">No training history was saved for this model.</p>

  const hovered = hover !== null ? chart.train[hover] : null
  // the validation point closest to where the pointer is
  const hoveredVal = hovered
    ? chart.val.reduce<{ step: number; loss: number } | null>(
        (best, p) => (!best || Math.abs(p.step - hovered.step) < Math.abs(best.step - hovered.step) ? p : best),
        null,
      )
    : null

  // finds the data point under the pointer or finger
  function onMove(e: React.PointerEvent<SVGSVGElement>) {
    if (!chart) return
    const box = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - box.left) / box.width) * W
    const step = ((px - PAD.left) / (W - PAD.left - PAD.right)) * chart.maxStep
    let nearest = 0
    for (let i = 1; i < chart.train.length; i++) {
      if (Math.abs(chart.train[i].step - step) < Math.abs(chart.train[nearest].step - step)) nearest = i
    }
    setHover(nearest)
  }

  const last = chart.train[chart.train.length - 1]
  const lastVal = chart.val.length ? chart.val[chart.val.length - 1] : null

  return (
    <figure className="grid gap-3">
      {/* legend and the numbers for the point under the pointer */}
      <figcaption className="flex flex-wrap items-center gap-x-5 gap-y-1 text-sm">
        <span className="flex items-center gap-2">
          <span className="h-0.5 w-5 bg-border-strong" aria-hidden="true" />
          Training loss
        </span>
        <span className="flex items-center gap-2">
          <span className="h-0.5 w-5 bg-accent" aria-hidden="true" />
          Validation loss
        </span>
        <span className="num ml-auto text-xs text-muted" aria-live="polite">
          {hovered
            ? `step ${hovered.step.toLocaleString()}: train ${hovered.loss.toFixed(3)}${hoveredVal ? `, val ${hoveredVal.loss.toFixed(3)}` : ''}`
            : `final: train ${last.loss.toFixed(3)}${lastVal ? `, val ${lastVal.loss.toFixed(3)}` : ''}`}
        </span>
      </figcaption>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full touch-pan-y select-none"
        role="img"
        aria-label={`Loss curve. Training loss ends at ${last.loss.toFixed(2)}${lastVal ? ` and validation loss at ${lastVal.loss.toFixed(2)}` : ''} after ${last.step.toLocaleString()} steps.`}
        onPointerMove={onMove}
        onPointerDown={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {/* horizontal guide lines with loss values */}
        {ticks(chart.bottom, chart.top, 4).map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={chart.y(t)} y2={chart.y(t)} stroke="var(--border)" />
            <text x={PAD.left - 8} y={chart.y(t) + 4} textAnchor="end" fontSize="11" fill="var(--muted)" className="num">
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        {/* step numbers along the bottom */}
        {ticks(0, chart.maxStep, 4).map((t) => (
          <text key={t} x={chart.x(t)} y={H - 12} textAnchor="middle" fontSize="11" fill="var(--muted)" className="num">
            {t >= 1000 ? `${(t / 1000).toFixed(t % 1000 ? 1 : 0)}k` : Math.round(t)}
          </text>
        ))}

        <path d={chart.trainPath} fill="none" stroke="var(--border-strong)" strokeWidth="1.5" strokeLinejoin="round" />
        <path d={chart.valPath} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />
        {/* dots on the validation line, it has far fewer points */}
        {chart.val.length <= 60 &&
          chart.val.map((p) => <circle key={p.step} cx={chart.x(p.step)} cy={chart.y(p.loss)} r="2.5" fill="var(--accent)" />)}

        {/* marker that follows the pointer */}
        {hovered && (
          <g>
            <line x1={chart.x(hovered.step)} x2={chart.x(hovered.step)} y1={PAD.top} y2={H - PAD.bottom} stroke="var(--text)" strokeOpacity="0.35" />
            <circle cx={chart.x(hovered.step)} cy={chart.y(hovered.loss)} r="3.5" fill="var(--text)" />
          </g>
        )}
      </svg>
      <p className="text-xs text-muted">Training steps on the horizontal axis, loss on the vertical axis. Lower is better.</p>
    </figure>
  )
}
