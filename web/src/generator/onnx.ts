import type { AttentionData, ModelConfig } from '../lib/types'
import type { WorkerRequest, WorkerResponse } from './messages'
import type { GeneratorInfo, LoadProgress, TextGenerator } from './types'

export interface OnnxOptions {
  forceBackend?: 'webgpu' | 'wasm'
  forcePrecision?: 'fp32' | 'int8'
}

type Pending = {
  resolve: (value: WorkerResponse) => void
  reject: (reason: Error) => void
  onProgress?: (p: LoadProgress) => void
}

// the real model: a thin wrapper that passes work to the web worker and waits for answers
export class OnnxGenerator implements TextGenerator {
  info: GeneratorInfo | null = null
  private worker: Worker | null = null
  private pending = new Map<number, Pending>()
  private nextId = 1
  private modelUrl: string
  private config: ModelConfig
  private options: OnnxOptions

  constructor(modelUrl: string, config: ModelConfig, options: OnnxOptions = {}) {
    this.modelUrl = modelUrl
    this.config = config
    this.options = options
    // models that don't export attention just don't get the method
    if (!config.has_attention) this.attention = undefined
  }

  // sends one request and resolves when the worker answers it
  private request(message: WorkerRequest, onProgress?: (p: LoadProgress) => void): Promise<WorkerResponse> {
    return new Promise((resolve, reject) => {
      if (!this.worker) return reject(new Error('Model is not loaded yet.'))
      this.pending.set(message.requestId, { resolve, reject, onProgress })
      this.worker.postMessage(message)
    })
  }

  private handle(response: WorkerResponse) {
    const entry = this.pending.get(response.requestId)
    if (!entry) return
    // progress messages don't finish the request
    if (response.type === 'progress') {
      entry.onProgress?.({ stage: response.stage, loaded: response.loaded, total: response.total })
      return
    }
    this.pending.delete(response.requestId)
    if (response.type === 'error') entry.reject(new Error(response.message))
    else entry.resolve(response)
  }

  async load(onProgress?: (p: LoadProgress) => void): Promise<void> {
    if (typeof Worker === 'undefined') throw new Error('This browser does not support web workers.')
    this.worker = new Worker(new URL('./onnx.worker.ts', import.meta.url), { type: 'module' })
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handle(event.data)
    // a crash inside the worker fails everything that was waiting on it
    this.worker.onerror = (event) => {
      for (const entry of this.pending.values()) entry.reject(new Error(event.message || 'The model worker crashed.'))
      this.pending.clear()
    }

    const response = await this.request(
      {
        type: 'load',
        requestId: this.nextId++,
        modelUrl: this.modelUrl,
        files: this.config.files,
        ...this.options,
      },
      onProgress,
    )
    if (response.type === 'loaded') this.info = { backend: response.backend, precision: response.precision }
  }

  async nextLogits(ids: number[]): Promise<Float32Array> {
    const response = await this.request({ type: 'run', requestId: this.nextId++, ids, wantAttention: false })
    if (response.type !== 'result') throw new Error('Unexpected answer from the model worker.')
    return response.logits
  }

  attention?: (ids: number[]) => Promise<AttentionData> = async (ids) => {
    const response = await this.request({ type: 'run', requestId: this.nextId++, ids, wantAttention: true })
    if (response.type !== 'result' || !response.attention) throw new Error('This model did not return attention.')
    return response.attention
  }

  dispose(): void {
    this.worker?.terminate()
    this.worker = null
    this.pending.clear()
    this.info = null
  }
}
