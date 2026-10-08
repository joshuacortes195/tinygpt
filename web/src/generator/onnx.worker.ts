// runs the model off the main thread so the page never freezes while it thinks
import * as ort from 'onnxruntime-web/webgpu'
import { fetchBytes } from '../lib/download'
import type { ModelFile } from '../lib/types'
import type { WorkerRequest, WorkerResponse } from './messages'

let session: ort.InferenceSession | null = null
// requests are handled one after another, a session can't run two things at once
let queue: Promise<void> = Promise.resolve()

function send(message: WorkerResponse, transfer: Transferable[] = []) {
  ;(self as unknown as Worker).postMessage(message, transfer)
}

// true when this browser can actually hand us a gpu
async function hasWebGPU(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu
  if (!gpu) return false
  try {
    return (await gpu.requestAdapter()) !== null
  } catch {
    return false
  }
}

// the list of urls behind a model file, one entry unless it was split into parts
function fileUrls(file: ModelFile, modelUrl: string): string[] {
  const names = file.parts ?? (file.file ? [file.file] : [])
  return names.map((name) => new URL(name, modelUrl).href)
}

async function load(req: Extract<WorkerRequest, { type: 'load' }>) {
  // plan: gpu with full precision if we can, otherwise cpu with the small int8 file
  const attempts: { backend: 'webgpu' | 'wasm'; precision: 'fp32' | 'int8' }[] = []
  const gpuOk = req.forceBackend !== 'wasm' && (await hasWebGPU())
  if (gpuOk && req.files.fp32) attempts.push({ backend: 'webgpu', precision: 'fp32' })
  if (req.forceBackend !== 'webgpu') {
    const cpuPrecision = req.forcePrecision ?? (req.files.int8 ? 'int8' : 'fp32')
    if (req.files[cpuPrecision]) attempts.push({ backend: 'wasm', precision: cpuPrecision })
  }
  if (!attempts.length) throw new Error('This browser has no backend that can run the model.')

  // downloaded files are kept so a fallback doesn't fetch the same thing twice
  const downloaded = new Map<string, Uint8Array>()
  let lastError: unknown = null

  for (const attempt of attempts) {
    try {
      const file = req.files[attempt.precision]!
      let bytes = downloaded.get(attempt.precision)
      if (!bytes) {
        bytes = await fetchBytes(fileUrls(file, req.modelUrl), file.bytes, (loaded, total) =>
          send({ type: 'progress', requestId: req.requestId, stage: 'download', loaded, total }),
        )
        downloaded.set(attempt.precision, bytes)
      }
      send({ type: 'progress', requestId: req.requestId, stage: 'start', loaded: 1, total: 1 })
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: [attempt.backend],
        graphOptimizationLevel: 'all',
      })
      // one tiny run to make sure the backend really works before we say it loaded
      await runSession([0], false)
      send({ type: 'loaded', requestId: req.requestId, backend: attempt.backend, precision: attempt.precision })
      return
    } catch (err) {
      // this backend didn't work out, try the next one
      lastError = err
      session = null
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError))
}

async function runSession(ids: number[], wantAttention: boolean) {
  if (!session) throw new Error('Model is not loaded yet.')
  // onnx wants 64 bit ints for token ids
  const input = new ort.Tensor('int64', BigInt64Array.from(ids, (id) => BigInt(id)), [1, ids.length])
  // only ask for attention when the heatmap needs it, it is a lot of data
  const outputs = wantAttention ? ['logits', 'attention'] : ['logits']
  return session.run({ input_ids: input }, outputs)
}

async function run(req: Extract<WorkerRequest, { type: 'run' }>) {
  const result = await runSession(req.ids, req.wantAttention)
  // copy out of the runtime's memory so the buffers can be handed to the page
  const logits = new Float32Array(result.logits.data as Float32Array)
  const transfer: Transferable[] = [logits.buffer]

  let attention
  if (req.wantAttention && result.attention) {
    // dims are [batch, layers, heads, time, time]
    const dims = result.attention.dims
    const data = new Float32Array(result.attention.data as Float32Array)
    attention = { layers: Number(dims[1]), heads: Number(dims[2]), size: Number(dims[3]), data }
    transfer.push(data.buffer)
  }
  send({ type: 'result', requestId: req.requestId, logits, attention }, transfer)
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const req = event.data
  queue = queue
    .then(() => (req.type === 'load' ? load(req) : run(req)))
    .catch((err: unknown) => {
      const message = err instanceof Error ? err.message : String(err)
      send({ type: 'error', requestId: req.requestId, message })
    })
}
