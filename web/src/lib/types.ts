// one row in manifest.json
export interface ManifestEntry {
  id: string
  name: string
  description: string
  // folder name under models/, or a full url if the files live somewhere else
  path: string
}

export interface Manifest {
  default: string
  models: ManifestEntry[]
}

// a model file is either one file or several parts that get glued together
export interface ModelFile {
  file?: string
  parts?: string[]
  bytes: number
}

// config.json inside each model folder
export interface ModelConfig {
  name: string
  description: string
  vocab_size: number
  block_size: number
  n_layer: number
  n_head: number
  n_embd: number
  params: number
  training_tokens: number | null
  training_steps: number | null
  training_seconds: number | null
  final_val_loss: number | null
  has_attention: boolean
  files: { fp32?: ModelFile; int8?: ModelFile }
  parity?: Record<string, number>
}

// metrics.json inside each model folder
export interface Metrics {
  train: { step: number; loss: number; lr: number }[]
  val: { step: number; loss: number }[]
  tokens_per_sec: number | null
}

export type Backend = 'webgpu' | 'wasm' | 'mock'
export type Precision = 'fp32' | 'int8' | 'none'

// the sliders in the playground
export interface SamplingSettings {
  temperature: number
  topK: number
  topP: number
  maxTokens: number
  seed: number
}

export const DEFAULT_SETTINGS: SamplingSettings = {
  temperature: 0.8,
  topK: 50,
  topP: 0.95,
  maxTokens: 120,
  seed: 42,
}

// one of the model's top guesses for a step
export interface Candidate {
  id: number
  prob: number
}

// a token on screen, either typed by the visitor or written by the model
export interface TokenInfo {
  id: number
  fromPrompt: boolean
  // chance the model gave the token it picked, only for generated tokens
  prob?: number
  top?: Candidate[]
}

// attention weights for one run, flattened as [layer][head][row][col]
export interface AttentionData {
  layers: number
  heads: number
  size: number
  data: Float32Array
}
