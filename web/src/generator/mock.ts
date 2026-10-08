import type { AttentionData } from '../lib/types'
import type { GeneratorInfo, LoadProgress, TextGenerator } from './types'

// fake model for development and tests: instant, and always gives the same answer
export class MockGenerator implements TextGenerator {
  info: GeneratorInfo | null = null
  private vocabSize: number
  private script: number[]
  private withAttention: boolean

  // script is a list of token ids the mock "wants" to write, on repeat
  constructor(vocabSize: number, script: number[] = [], withAttention = true) {
    this.vocabSize = vocabSize
    this.script = script
    this.withAttention = withAttention
    // drop the method entirely to act like a model without attention
    if (!withAttention) this.attention = undefined
  }

  async load(onProgress?: (p: LoadProgress) => void): Promise<void> {
    onProgress?.({ stage: 'download', loaded: 1, total: 1 })
    this.info = { backend: 'mock', precision: 'none' }
  }

  async nextLogits(ids: number[]): Promise<Float32Array> {
    const logits = new Float32Array(this.vocabSize).fill(-4)
    // main guess: the next token of the script, or a simple function of the last token
    const last = ids.length ? ids[ids.length - 1] : 0
    const favorite = this.script.length
      ? this.script[ids.length % this.script.length]
      : (last * 31 + ids.length * 7 + 1) % this.vocabSize
    logits[favorite] = 6
    // two weaker guesses so the probability view has something to show
    logits[(favorite + 1) % this.vocabSize] = Math.max(logits[(favorite + 1) % this.vocabSize], 3)
    logits[(favorite + 2) % this.vocabSize] = Math.max(logits[(favorite + 2) % this.vocabSize], 2)
    return logits
  }

  attention?: (ids: number[]) => Promise<AttentionData> = async (ids) => {
    const size = ids.length
    const layers = 2
    const heads = 2
    const data = new Float32Array(layers * heads * size * size)
    // every token spreads its attention evenly over itself and the tokens before it
    for (let m = 0; m < layers * heads; m++) {
      for (let row = 0; row < size; row++) {
        for (let col = 0; col <= row; col++) {
          data[m * size * size + row * size + col] = 1 / (row + 1)
        }
      }
    }
    return { layers, heads, size, data }
  }

  dispose(): void {
    this.info = null
  }

  get hasAttention(): boolean {
    return this.withAttention
  }
}
