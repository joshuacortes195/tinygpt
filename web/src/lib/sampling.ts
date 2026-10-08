import type { Candidate, SamplingSettings } from './types'

// small seeded random number generator so the same seed gives the same story
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// turns raw scores into chances that add up to 1
export function softmax(logits: Float32Array, temperature = 1): Float32Array {
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i]
  const out = new Float32Array(logits.length)
  let sum = 0
  for (let i = 0; i < logits.length; i++) {
    const v = Math.exp((logits[i] - max) / temperature)
    out[i] = v
    sum += v
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum
  return out
}

// index of the biggest value
export function argmax(values: Float32Array): number {
  let best = 0
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i
  return best
}

// the n most likely tokens, best first
export function topCandidates(probs: Float32Array, n: number): Candidate[] {
  const top: Candidate[] = []
  for (let id = 0; id < probs.length; id++) {
    const prob = probs[id]
    if (top.length < n) {
      top.push({ id, prob })
      top.sort((a, b) => b.prob - a.prob)
    } else if (prob > top[n - 1].prob) {
      top[n - 1] = { id, prob }
      top.sort((a, b) => b.prob - a.prob)
    }
  }
  return top
}

// picks the next token, same rules as model/tinygpt/sampling.py
export function sampleToken(
  logits: Float32Array,
  settings: Pick<SamplingSettings, 'temperature' | 'topK' | 'topP'>,
  random: () => number,
): number {
  // temperature 0 means always take the most likely token
  if (settings.temperature <= 0) return argmax(logits)

  const probs = softmax(logits, settings.temperature)
  // sort ids from most to least likely
  const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a])

  // top-k: keep only the k best tokens
  let keep = settings.topK > 0 ? Math.min(settings.topK, order.length) : order.length

  // top-p: keep the smallest set of tokens whose chances add up to p
  if (settings.topP > 0 && settings.topP < 1) {
    let cumulative = 0
    for (let i = 0; i < keep; i++) {
      cumulative += probs[order[i]]
      if (cumulative >= settings.topP) {
        keep = i + 1
        break
      }
    }
  }

  // roll the dice among the tokens that are left
  let total = 0
  for (let i = 0; i < keep; i++) total += probs[order[i]]
  let r = random() * total
  for (let i = 0; i < keep; i++) {
    r -= probs[order[i]]
    if (r <= 0) return order[i]
  }
  return order[keep - 1]
}
