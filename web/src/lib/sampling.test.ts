import { describe, expect, it } from 'vitest'
import { argmax, mulberry32, sampleToken, softmax, topCandidates } from './sampling'

const logits = new Float32Array([5, 4, 1, 0, -1])

describe('sampling', () => {
  it('softmax adds up to 1 and keeps the order', () => {
    const probs = softmax(logits)
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5)
    expect(argmax(probs)).toBe(0)
    expect(probs[0]).toBeGreaterThan(probs[1])
  })

  it('lower temperature makes the top token more likely', () => {
    expect(softmax(logits, 0.5)[0]).toBeGreaterThan(softmax(logits, 1)[0])
  })

  it('temperature 0 is greedy', () => {
    const random = mulberry32(1)
    for (let i = 0; i < 20; i++) expect(sampleToken(logits, { temperature: 0, topK: 0, topP: 1 }, random)).toBe(0)
  })

  it('top-k only picks from the k best', () => {
    const random = mulberry32(2)
    const picks = new Set<number>()
    for (let i = 0; i < 300; i++) picks.add(sampleToken(logits, { temperature: 1.5, topK: 2, topP: 1 }, random))
    expect([...picks].sort()).toEqual([0, 1])
  })

  it('a tiny top-p keeps just the best token', () => {
    const random = mulberry32(3)
    for (let i = 0; i < 50; i++) expect(sampleToken(logits, { temperature: 1, topK: 0, topP: 0.1 }, random)).toBe(0)
  })

  it('with everything off it can reach unlikely tokens', () => {
    const random = mulberry32(4)
    const picks = new Set<number>()
    for (let i = 0; i < 3000; i++) picks.add(sampleToken(logits, { temperature: 1.5, topK: 0, topP: 1 }, random))
    expect(picks.size).toBe(5)
  })

  it('the same seed gives the same picks', () => {
    const run = (seed: number) => {
      const random = mulberry32(seed)
      return Array.from({ length: 30 }, () => sampleToken(logits, { temperature: 1, topK: 0, topP: 1 }, random))
    }
    expect(run(7)).toEqual(run(7))
    expect(run(7)).not.toEqual(run(8))
  })

  it('topCandidates returns the best n in order', () => {
    const top = topCandidates(softmax(logits), 3)
    expect(top.map((c) => c.id)).toEqual([0, 1, 2])
    expect(top[0].prob).toBeGreaterThan(top[1].prob)
  })
})
