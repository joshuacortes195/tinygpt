import { EOT } from '../generator/generate'
import { MockGenerator } from '../generator/mock'
import { OnnxGenerator, type OnnxOptions } from '../generator/onnx'
import type { LoadProgress, TextGenerator } from '../generator/types'
import { BPETokenizer, type TokenizerData } from '../tokenizer/tokenizer'
import { loadConfig, modelUrl } from './manifest'
import type { ManifestEntry, ModelConfig } from './types'

// everything the ui needs to use one model
export interface LoadedModel {
  entry: ManifestEntry
  config: ModelConfig
  tokenizer: BPETokenizer
  generator: TextGenerator
}

// the fake model, add ?mock=1 to the url to get it in the picker
export const MOCK_ENTRY: ManifestEntry = {
  id: 'mock',
  name: 'Mock model',
  description: 'Fake model for development. Instant and always the same.',
  path: 'mock',
}

const MOCK_STORY = ' Once upon a time there was a tiny model that only knew one story, so it told it again.'

function loadMock(): LoadedModel {
  // byte level tokenizer with no merges, so every character is its own token
  const data: TokenizerData = {
    version: 1,
    pattern: "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\p{White_Space}\\p{L}\\p{N}]+|\\p{White_Space}+",
    merges: [],
    special_tokens: { [EOT]: 256 },
  }
  const tokenizer = new BPETokenizer(data)
  const config: ModelConfig = {
    name: MOCK_ENTRY.name,
    description: MOCK_ENTRY.description,
    vocab_size: 257,
    block_size: 64,
    n_layer: 2,
    n_head: 2,
    n_embd: 0,
    params: 0,
    training_tokens: null,
    training_steps: null,
    training_seconds: null,
    final_val_loss: null,
    has_attention: true,
    files: {},
  }
  return { entry: MOCK_ENTRY, config, tokenizer, generator: new MockGenerator(257, tokenizer.encode(MOCK_STORY)) }
}

// ?backend=wasm&precision=fp32 forces a backend, handy for benchmarking
function optionsFromUrl(): OnnxOptions {
  const params = new URLSearchParams(window.location.search)
  const backend = params.get('backend')
  const precision = params.get('precision')
  return {
    forceBackend: backend === 'webgpu' || backend === 'wasm' ? backend : undefined,
    forcePrecision: precision === 'fp32' || precision === 'int8' ? precision : undefined,
  }
}

async function loadReal(entry: ManifestEntry, onProgress?: (p: LoadProgress) => void): Promise<LoadedModel> {
  const base = modelUrl(entry)
  // config and tokenizer are small, grab them together
  const [config, tokenizerData] = await Promise.all([
    loadConfig(entry),
    fetch(new URL('tokenizer.json', base).href).then((res) => {
      if (!res.ok) throw new Error(`Could not load the tokenizer (${res.status})`)
      return res.json() as Promise<TokenizerData>
    }),
  ])
  const generator = new OnnxGenerator(base, config, optionsFromUrl())
  await generator.load(onProgress)
  return { entry, config, tokenizer: new BPETokenizer(tokenizerData), generator }
}

// models stay loaded while you move between pages
const cache = new Map<string, Promise<LoadedModel>>()

export function getModel(entry: ManifestEntry, onProgress?: (p: LoadProgress) => void): Promise<LoadedModel> {
  let promise = cache.get(entry.id)
  if (!promise) {
    promise = entry.id === MOCK_ENTRY.id ? Promise.resolve().then(loadMock) : loadReal(entry, onProgress)
    // a failed load should be retryable, so don't keep it
    promise.catch(() => cache.delete(entry.id))
    cache.set(entry.id, promise)
  }
  return promise
}
