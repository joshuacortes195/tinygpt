import type { Backend, ModelFile, Precision } from '../lib/types'

// messages the page sends to the worker
export type WorkerRequest =
  | {
      type: 'load'
      requestId: number
      modelUrl: string
      files: { fp32?: ModelFile; int8?: ModelFile }
      // optional overrides, used for benchmarking one backend against another
      forceBackend?: 'webgpu' | 'wasm'
      forcePrecision?: 'fp32' | 'int8'
    }
  | { type: 'run'; requestId: number; ids: number[]; wantAttention: boolean }

// messages the worker sends back
export type WorkerResponse =
  | { type: 'progress'; requestId: number; stage: 'download' | 'start'; loaded: number; total: number }
  | { type: 'loaded'; requestId: number; backend: Backend; precision: Precision }
  | {
      type: 'result'
      requestId: number
      logits: Float32Array
      attention?: { layers: number; heads: number; size: number; data: Float32Array }
    }
  | { type: 'error'; requestId: number; message: string }
