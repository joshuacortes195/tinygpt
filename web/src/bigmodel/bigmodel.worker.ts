// runs the optional fine-tuned model with transformers.js, off the main thread
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  env,
  InterruptableStoppingCriteria,
  TextStreamer,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from '@huggingface/transformers'

export type BigRequest =
  | { type: 'load'; modelsUrl: string; id: string; dtype: string; bytes: number }
  | { type: 'run'; prompt: string; maxTokens: number; temperature: number; topK: number; topP: number }
  | { type: 'stop' }

export type BigResponse =
  | { type: 'progress'; loaded: number; total: number }
  | { type: 'loaded' }
  | { type: 'text'; text: string }
  | { type: 'done'; tokens: number; seconds: number }
  | { type: 'error'; message: string }

let tokenizer: PreTrainedTokenizer | null = null
let model: PreTrainedModel | null = null
const stopper = new InterruptableStoppingCriteria()

function send(message: BigResponse) {
  ;(self as unknown as Worker).postMessage(message)
}

async function load(req: Extract<BigRequest, { type: 'load' }>) {
  // only ever read model files from this site, never from the hub
  env.allowRemoteModels = false
  env.allowLocalModels = true
  env.localModelPath = req.modelsUrl

  // the host gzips model files, so the size header is the zipped size and the library crawls
  // trying to grow its buffer. read each file here and hand it over with its real size
  let loaded = 0
  env.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const res = await fetch(input, init)
    const partial = new Headers(init?.headers).has('range')
    if (!res.ok || !res.body || partial) return res
    // collect the pieces and report how far along the whole download is
    const reader = res.body.getReader()
    const chunks: BlobPart[] = []
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      loaded += value.length
      send({ type: 'progress', loaded: Math.min(loaded, req.bytes), total: req.bytes })
    }
    // same response, now with the right length on it
    const blob = new Blob(chunks)
    const headers = new Headers(res.headers)
    headers.set('content-length', String(blob.size))
    headers.delete('content-encoding')
    return new Response(blob, { status: res.status, statusText: res.statusText, headers })
  }

  tokenizer = await AutoTokenizer.from_pretrained(req.id)
  model = await AutoModelForCausalLM.from_pretrained(req.id, { dtype: req.dtype as 'q8', device: 'wasm' })
  send({ type: 'loaded' })
}

async function run(req: Extract<BigRequest, { type: 'run' }>) {
  if (!tokenizer || !model) throw new Error('The model is not loaded yet.')
  stopper.reset()
  let tokens = 0
  const start = performance.now()
  // sends each piece of text to the page as soon as it is decoded
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text: string) => send({ type: 'text', text }),
    token_callback_function: () => {
      tokens += 1
    },
  })
  // the prompt as token ids, then let the model keep writing
  const inputs = tokenizer(req.prompt)
  await model.generate({
    ...inputs,
    max_new_tokens: req.maxTokens,
    do_sample: req.temperature > 0,
    temperature: req.temperature > 0 ? req.temperature : 1,
    top_k: req.topK > 0 ? req.topK : 0,
    top_p: req.topP,
    streamer,
    stopping_criteria: stopper,
  })
  send({ type: 'done', tokens, seconds: (performance.now() - start) / 1000 })
}

self.onmessage = (event: MessageEvent<BigRequest>) => {
  const req = event.data
  // stop has to cut in while a run is still going
  if (req.type === 'stop') {
    stopper.interrupt()
    return
  }
  const job = req.type === 'load' ? load(req) : run(req)
  job.catch((err: unknown) => {
    console.error('bigmodel', err)
    send({ type: 'error', message: (err instanceof Error ? err.message : String(err)) || 'Unknown error' })
  })
}
