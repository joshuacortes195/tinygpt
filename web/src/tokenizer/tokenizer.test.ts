import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { BPETokenizer, type TokenizerData } from './tokenizer'

// the same files the python tests read
const fixtures = resolve(__dirname, '../../../shared/fixtures')
const data: TokenizerData = JSON.parse(readFileSync(resolve(fixtures, 'tokenizer.json'), 'utf-8'))
const golden: { vocab_size: number; cases: { name: string; text: string; ids: number[] }[] } = JSON.parse(
  readFileSync(resolve(fixtures, 'tokenizer_cases.json'), 'utf-8'),
)

describe('BPETokenizer', () => {
  const tok = new BPETokenizer(data)

  it('has the same vocab size as python', () => {
    expect(tok.vocabSize).toBe(golden.vocab_size)
  })

  // every golden case must give the exact ids python produced
  for (const c of golden.cases) {
    it(`matches python on: ${c.name}`, () => {
      expect(tok.encode(c.text)).toEqual(c.ids)
      expect(tok.decode(c.ids)).toBe(c.text)
    })
  }

  it('treats special tokens as plain text when asked', () => {
    const name = Object.keys(data.special_tokens)[0]
    const ids = tok.encode(name, false)
    expect(ids).not.toContain(data.special_tokens[name])
    expect(tok.decode(ids)).toBe(name)
  })

  it('round trips text it has never seen', () => {
    const text = 'Zxqv ⟁ ∑ ☃ 🦖 końcówka'
    expect(tok.decode(tok.encode(text))).toBe(text)
  })
})
