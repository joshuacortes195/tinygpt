import { useCallback, useEffect, useRef, useState } from 'react'
import { generate, type GenerateStats } from '../generator/generate'
import type { LoadedModel } from '../lib/models'
import type { SamplingSettings, TokenInfo } from '../lib/types'

// runs the model and keeps the list of tokens that are on screen
export function useGeneration(model: LoadedModel | null) {
  const [tokens, setTokens] = useState<TokenInfo[]>([])
  const [running, setRunning] = useState(false)
  const [stats, setStats] = useState<GenerateStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const stop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  // switching models throws away the old output, the token ids wouldn't mean the same thing
  useEffect(() => {
    abortRef.current?.abort()
    setTokens([])
    setStats(null)
    setError(null)
  }, [model])

  const run = useCallback(
    async (prompt: string, settings: SamplingSettings) => {
      if (!model) return null
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      // show the prompt as tokens right away
      const promptIds = model.tokenizer.encode(prompt, false)
      setTokens(promptIds.map((id) => ({ id, fromPrompt: true })))
      setStats(null)
      setError(null)
      setRunning(true)

      try {
        const result = await generate({
          generator: model.generator,
          tokenizer: model.tokenizer,
          prompt,
          settings,
          blockSize: model.config.block_size,
          signal: controller.signal,
          // adds each new token to the screen as it arrives
          onToken: (token) => {
            if (!controller.signal.aborted) setTokens((prev) => [...prev, token])
          },
        })
        if (abortRef.current === controller) setStats(result)
        return result
      } catch (err) {
        if (abortRef.current === controller) setError(err instanceof Error ? err.message : String(err))
        return null
      } finally {
        if (abortRef.current === controller) setRunning(false)
      }
    },
    [model],
  )

  return { tokens, running, stats, error, run, stop }
}
