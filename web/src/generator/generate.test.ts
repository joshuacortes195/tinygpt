import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { DEFAULT_SETTINGS, type TokenInfo } from '../lib/types'
import { BPETokenizer, type TokenizerData } from '../tokenizer/tokenizer'
import { EOT, generate, promptTokens } from './generate'
import { MockGenerator } from './mock'

// the real tokenizer file, the model itself is faked
const data: TokenizerData = JSON.parse(
  readFileSync(resolve(__dirname, '../../../shared/fixtures/tokenizer.json'), 'utf-8'),
)
const tokenizer = new BPETokenizer(data)
const settings = { ...DEFAULT_SETTINGS, maxTokens: 12 }

async function collect(overrides: Partial<Parameters<typeof generate>[0]> = {}) {
  const tokens: TokenInfo[] = []
  const generator = new MockGenerator(tokenizer.vocabSize)
  await generator.load()
  const stats = await generate({
    generator,
    tokenizer,
    prompt: 'Once upon a time',
    settings,
    blockSize: 64,
    onToken: (t) => tokens.push(t),
    ...overrides,
  })
  return { tokens, stats }
}

describe('generate', () => {
  it('writes max tokens and reports each one', async () => {
    const { tokens, stats } = await collect()
    expect(tokens).toHaveLength(12)
    expect(stats.tokens).toBe(12)
    expect(stats.stopped).toBe('max')
    expect(tokens.every((t) => !t.fromPrompt)).toBe(true)
  })

  it('every token comes with its probability and the top 10 guesses', async () => {
    const { tokens } = await collect()
    for (const t of tokens) {
      expect(t.top).toHaveLength(10)
      expect(t.prob).toBeGreaterThan(0)
      expect(t.top![0].prob).toBeGreaterThanOrEqual(t.top![1].prob)
    }
  })

  it('is the same for the same seed and different for another', async () => {
    const hot = { ...settings, temperature: 1.5, topK: 0, topP: 1 }
    const a = await collect({ settings: { ...hot, seed: 1 } })
    const b = await collect({ settings: { ...hot, seed: 1 } })
    const c = await collect({ settings: { ...hot, seed: 2 } })
    const ids = (r: { tokens: TokenInfo[] }) => r.tokens.map((t) => t.id)
    expect(ids(a)).toEqual(ids(b))
    expect(ids(a)).not.toEqual(ids(c))
  })

  it('stops when asked', async () => {
    const controller = new AbortController()
    const tokens: TokenInfo[] = []
    const generator = new MockGenerator(tokenizer.vocabSize)
    const stats = await generate({
      generator,
      tokenizer,
      prompt: 'Hello',
      settings: { ...settings, maxTokens: 100 },
      blockSize: 64,
      signal: controller.signal,
      onToken: (t) => {
        tokens.push(t)
        if (tokens.length === 3) controller.abort()
      },
    })
    expect(stats.stopped).toBe('abort')
    expect(tokens).toHaveLength(3)
  })

  it('stops when the model writes the end-of-story token', async () => {
    const eot = tokenizer.specialTokens[EOT]
    // a mock that wants to write two tokens and then end the story
    const generator = new MockGenerator(tokenizer.vocabSize, [10, 11, 12, eot])
    const tokens: TokenInfo[] = []
    const stats = await generate({
      generator,
      tokenizer,
      prompt: 'a',
      settings: { ...settings, temperature: 0 },
      blockSize: 64,
      onToken: (t) => tokens.push(t),
    })
    expect(stats.stopped).toBe('end')
    expect(tokens.map((t) => t.id)).toEqual([11, 12])
  })

  it('never shows the model more than its context window', async () => {
    const lengths: number[] = []
    const generator = new MockGenerator(tokenizer.vocabSize)
    const original = generator.nextLogits.bind(generator)
    generator.nextLogits = (ids) => {
      lengths.push(ids.length)
      return original(ids)
    }
    await generate({ generator, tokenizer, prompt: 'Once upon a time', settings, blockSize: 6, onToken: () => {} })
    expect(Math.max(...lengths)).toBe(6)
  })

  it('starts an empty prompt from the end-of-story token', () => {
    expect(promptTokens(tokenizer, '')).toEqual([tokenizer.specialTokens[EOT]])
    expect(promptTokens(tokenizer, 'Hi').length).toBeGreaterThan(0)
  })
})

describe('MockGenerator', () => {
  it('gives the same scores for the same input', async () => {
    const gen = new MockGenerator(100)
    expect(await gen.nextLogits([1, 2, 3])).toEqual(await gen.nextLogits([1, 2, 3]))
  })

  it('returns attention rows that add up to 1 and never look ahead', async () => {
    const gen = new MockGenerator(100)
    const a = await gen.attention!([1, 2, 3, 4])
    expect(a.data).toHaveLength(a.layers * a.heads * 16)
    const row = Array.from(a.data.subarray(4, 8))
    expect(row.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 5)
    expect(row[2]).toBe(0)
  })

  it('can act like a model without attention', () => {
    expect(new MockGenerator(100, [], false).attention).toBeUndefined()
  })
})
