import type { Manifest, ManifestEntry, Metrics, ModelConfig } from './types'

// where the models folder lives, relative to wherever the site is hosted
export function modelsBaseUrl(): string {
  return new URL('models/', document.baseURI).href
}

// folder url for one model, the path can also be a full url on another host
export function modelUrl(entry: ManifestEntry): string {
  const path = entry.path.endsWith('/') ? entry.path : `${entry.path}/`
  return new URL(path, modelsBaseUrl()).href
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not load ${url} (${res.status})`)
  return (await res.json()) as T
}

// the list of models the site knows about
export function loadManifest(): Promise<Manifest> {
  return getJson<Manifest>(new URL('manifest.json', modelsBaseUrl()).href)
}

export function loadConfig(entry: ManifestEntry): Promise<ModelConfig> {
  return getJson<ModelConfig>(new URL('config.json', modelUrl(entry)).href)
}

export function loadMetrics(entry: ManifestEntry): Promise<Metrics> {
  return getJson<Metrics>(new URL('metrics.json', modelUrl(entry)).href)
}
