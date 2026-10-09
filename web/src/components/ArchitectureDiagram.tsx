import { ArrowBendDownLeftIcon, ArrowRightIcon } from '@phosphor-icons/react'
import { formatCount } from '../lib/format'
import type { ModelConfig } from '../lib/types'

// the path one piece of text takes through the model, left to right, with this model's real sizes
export function ArchitectureDiagram({ config }: { config: ModelConfig }) {
  const vocab = config.vocab_size.toLocaleString()
  return (
    <div>
      {/* on small screens the chart keeps its width and scrolls sideways inside this box */}
      <div className="overflow-x-auto pb-2" tabIndex={0} role="group" aria-label="Model architecture, scrolls sideways">
        <div className="min-w-[66rem]">
          <ol className="flex items-stretch" aria-label="Model architecture, from input to output">
            <Step n="01" title="Your text" detail={'"Once upon a time"'} />
            <Arrow />
            <Step n="02" title="Tokenizer" detail={`Text is cut into ids from a vocab of ${vocab}`} />
            <Arrow />
            <Step
              n="03"
              title="Embeddings"
              detail={`Each id becomes ${config.n_embd} numbers, plus its position (up to ${config.block_size})`}
            />
            <Arrow />

            {/* the part that repeats */}
            <li className="flex flex-[2.5] basis-0 flex-col rounded-[4px] border border-dashed border-border-strong p-2">
              <p className="num mb-2 flex items-baseline justify-between gap-3 px-1 text-xs">
                <span className="text-text">Transformer block</span>
                <span className="text-accent">x {config.n_layer}</span>
              </p>
              <ol className="flex flex-1 items-stretch">
                <Step
                  n="04"
                  title="Self-attention"
                  detail={`${config.n_head} heads. Each token looks back at earlier ones, never ahead`}
                  accent
                />
                <Arrow />
                <Step n="05" title="MLP" detail={`Widen to ${config.n_embd * 4}, GELU, back to ${config.n_embd}`} />
              </ol>
              <p className="mt-2 px-1 text-xs leading-snug text-muted">
                LayerNorm before each step, and each result is added back to its input.
              </p>
            </li>

            <Arrow />
            <Step n="06" title="Output layer" detail="A last LayerNorm, then weights shared with the embedding table" />
            <Arrow />
            <Step n="07" title="Next token" detail={`A score for each of the ${vocab} tokens. One gets sampled`} />
          </ol>

          {/* the line that loops back to the start */}
          <div className="mx-[6%] flex h-9 items-end justify-center rounded-b-[4px] border border-t-0 border-dashed border-border-strong">
            <p className="num flex translate-y-1/2 items-center gap-2 bg-bg px-3 text-xs text-muted">
              <ArrowBendDownLeftIcon size={14} className="text-accent" />
              the new token is added to the text and it all runs again
            </p>
          </div>
        </div>
      </div>
      <p className="num mt-5 text-xs text-muted">
        {formatCount(config.params)} parameters in total
        <span className="lg:hidden">. Swipe sideways to follow the whole path.</span>
      </p>
    </div>
  )
}

function Step({ n, title, detail, accent }: { n: string; title: string; detail: string; accent?: boolean }) {
  return (
    <li
      className={`flex-1 basis-0 rounded-[4px] border p-3 ${
        accent ? 'border-accent bg-accent-soft' : 'border-border bg-surface'
      }`}
    >
      <p className={`num text-[0.6875rem] ${accent ? 'text-text' : 'text-accent'}`}>{n}</p>
      <p className="mt-1 text-[0.9375rem] leading-snug font-medium">{title}</p>
      <p className={`mt-1.5 text-[0.8125rem] leading-snug ${accent ? 'text-text' : 'text-muted'}`}>{detail}</p>
    </li>
  )
}

function Arrow() {
  return (
    <li aria-hidden="true" className="flex w-6 shrink-0 items-center justify-center text-muted">
      <ArrowRightIcon size={14} />
    </li>
  )
}
