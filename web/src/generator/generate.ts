import { mulberry32, sampleToken, softmax, topCandidates } from '../lib/sampling'
import type { SamplingSettings, TokenInfo } from '../lib/types'
import type { BPETokenizer } from '../tokenizer/tokenizer'
import type { TextGenerator } from './types'

export const EOT = '<|endoftext|>'

export interface GenerateOptions {
  generator: TextGenerator
  tokenizer: BPETokenizer
  prompt: string
  settings: SamplingSettings
  // how many tokens the model can see at once
  blockSize: number
  onToken: (token: TokenInfo) => void
  signal?: AbortSignal
}

export interface GenerateStats {
  tokens: number
  seconds: number
  tokensPerSecond: number
  stopped: 'max' | 'end' | 'abort'
}

// the prompt as tokens, an empty prompt starts from the end-of-text marker
export function promptTokens(tokenizer: BPETokenizer, prompt: string): number[] {
  const ids = tokenizer.encode(prompt, false)
  if (ids.length) return ids
  const eot = tokenizer.specialTokens[EOT]
  return eot === undefined ? [0] : [eot]
}

// writes tokens one at a time and reports each one as soon as it is picked
export async function generate(options: GenerateOptions): Promise<GenerateStats> {
  const { generator, tokenizer, settings, blockSize, onToken, signal } = options
  const random = mulberry32(settings.seed)
  const eot = tokenizer.specialTokens[EOT]
  const ids = promptTokens(tokenizer, options.prompt)

  const start = performance.now()
  let count = 0
  let stopped: GenerateStats['stopped'] = 'max'

  while (count < settings.maxTokens) {
    if (signal?.aborted) {
      stopped = 'abort'
      break
    }
    // the model only sees the most recent blockSize tokens
    const logits = await generator.nextLogits(ids.slice(-blockSize))
    if (signal?.aborted) {
      stopped = 'abort'
      break
    }

    const id = sampleToken(logits, settings, random)
    // the model saying "this text is over"
    if (id === eot) {
      stopped = 'end'
      break
    }

    // what the model believed before any temperature or filtering, for the probability view
    const probs = softmax(logits)
    ids.push(id)
    count += 1
    onToken({ id, fromPrompt: false, prob: probs[id], top: topCandidates(probs, 10) })
  }

  const seconds = (performance.now() - start) / 1000
  return { tokens: count, seconds, tokensPerSecond: seconds > 0 ? count / seconds : 0, stopped }
}
