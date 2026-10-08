import { ArrowDownIcon } from '@phosphor-icons/react'
import { formatCount } from '../lib/format'
import type { ModelConfig } from '../lib/types'

// the path one piece of text takes through the model, with this model's real sizes
export function ArchitectureDiagram({ config }: { config: ModelConfig }) {
  return (
    <ol className="mx-auto grid max-w-md gap-2" aria-label="Model architecture, from input to output">
      <Step title="Your text" detail={'"Once upon a time"'} plain />
      <Arrow />
      <Step title="BPE tokenizer" detail={`Text becomes ids from a vocab of ${config.vocab_size.toLocaleString()}`} />
      <Arrow />
      <Step
        title="Token + position embeddings"
        detail={`Each id becomes a vector of ${config.n_embd} numbers, plus one for where it sits (up to ${config.block_size})`}
      />
      <Arrow />

      {/* the part that repeats */}
      <li className="rounded-xl border border-dashed border-border-strong p-3">
        <p className="mb-2 flex items-baseline justify-between gap-3 text-sm">
          <span className="font-semibold">Transformer block</span>
          <span className="num text-muted">repeated {config.n_layer} times</span>
        </p>
        <ol className="grid gap-2">
          <Step title="LayerNorm" small />
          <Step
            title="Causal self-attention"
            detail={`${config.n_head} heads. Each token looks back at earlier tokens, never ahead`}
            accent
          />
          <Step title="Add back to the input" small />
          <Step title="LayerNorm" small />
          <Step title="MLP" detail={`Widen to ${config.n_embd * 4}, GELU, narrow back to ${config.n_embd}`} />
          <Step title="Add back to the input" small />
        </ol>
      </li>

      <Arrow />
      <Step title="Final LayerNorm" small />
      <Arrow />
      <Step title="Output layer" detail="Shares its weights with the token embedding table" />
      <Arrow />
      <Step
        title="Next token probabilities"
        detail={`One score for each of the ${config.vocab_size.toLocaleString()} tokens. Sample one, add it to the text, repeat`}
        plain
      />
      <li className="num pt-2 text-center text-sm text-muted">{formatCount(config.params)} parameters in total</li>
    </ol>
  )
}

function Step({
  title,
  detail,
  small,
  accent,
  plain,
}: {
  title: string
  detail?: string
  small?: boolean
  accent?: boolean
  plain?: boolean
}) {
  return (
    <li
      className={`rounded-lg border px-3 ${small ? 'py-1.5' : 'py-2.5'} ${
        accent
          ? 'border-accent bg-accent-soft'
          : plain
            ? 'border-transparent bg-surface-2'
            : 'border-border bg-surface'
      }`}
    >
      <p className={small ? 'text-sm text-muted' : 'text-[0.9375rem] font-medium'}>{title}</p>
      {detail && <p className="mt-0.5 text-sm leading-snug text-muted">{detail}</p>}
    </li>
  )
}

function Arrow() {
  return (
    <li aria-hidden="true" className="flex justify-center text-muted">
      <ArrowDownIcon size={16} />
    </li>
  )
}
