import type { AttentionData, Backend, Precision } from '../lib/types'

// progress while a model is being fetched and started
export interface LoadProgress {
  stage: 'download' | 'start'
  loaded: number
  total: number
}

export interface GeneratorInfo {
  backend: Backend
  precision: Precision
}

// anything that can turn token ids into scores for the next token
// the ui only talks to this, so a mock and the real onnx model are swappable
export interface TextGenerator {
  // which backend ended up being used, known after load()
  info: GeneratorInfo | null
  load(onProgress?: (p: LoadProgress) => void): Promise<void>
  // one score per vocab entry for the token that comes after ids
  nextLogits(ids: number[]): Promise<Float32Array>
  // attention maps for ids, missing when the model does not export them
  attention?(ids: number[]): Promise<AttentionData>
  dispose(): void
}
