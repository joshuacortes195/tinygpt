import { useEffect, useState } from 'react'
import type { LoadProgress } from '../generator/types'
import { getModel, type LoadedModel } from '../lib/models'
import type { ManifestEntry } from '../lib/types'

export type ModelState =
  | { status: 'idle' }
  | { status: 'loading'; progress: LoadProgress | null }
  | { status: 'ready'; model: LoadedModel }
  | { status: 'error'; message: string }

// loads the picked model and tells the ui how far along it is
export function useModel(entry: ManifestEntry | null, retryKey = 0): ModelState {
  const [state, setState] = useState<ModelState>({ status: 'idle' })

  useEffect(() => {
    if (!entry) {
      setState({ status: 'idle' })
      return
    }
    // ignore answers from a model the visitor already switched away from
    let cancelled = false
    setState({ status: 'loading', progress: null })
    getModel(entry, (progress) => {
      if (!cancelled) setState({ status: 'loading', progress })
    })
      .then((model) => {
        if (!cancelled) setState({ status: 'ready', model })
      })
      .catch((err: unknown) => {
        if (!cancelled) setState({ status: 'error', message: err instanceof Error ? err.message : String(err) })
      })
    return () => {
      cancelled = true
    }
  }, [entry, retryKey])

  return state
}
