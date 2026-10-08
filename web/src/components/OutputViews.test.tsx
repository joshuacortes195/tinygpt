import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { generate } from '../generator/generate'
import { MockGenerator } from '../generator/mock'
import type { LoadedModel } from '../lib/models'
import { DEFAULT_SETTINGS, type ModelConfig, type TokenInfo } from '../lib/types'
import { BPETokenizer, type TokenizerData } from '../tokenizer/tokenizer'
import { AttentionView, ProbabilityView, TextView, TokenView } from './OutputViews'

const data: TokenizerData = JSON.parse(
  readFileSync(resolve(__dirname, '../../../shared/fixtures/tokenizer.json'), 'utf-8'),
)
const tokenizer = new BPETokenizer(data)

// a loaded model backed by the mock, with or without attention
function mockModel(withAttention = true): LoadedModel {
  const config = { block_size: 64, n_layer: 2, n_head: 2, has_attention: withAttention } as ModelConfig
  return {
    entry: { id: 'mock', name: 'Mock', description: '', path: 'mock' },
    config,
    tokenizer,
    generator: new MockGenerator(tokenizer.vocabSize, [], withAttention),
  }
}

// prompt tokens followed by a few generated ones
async function someTokens(model: LoadedModel): Promise<TokenInfo[]> {
  const tokens: TokenInfo[] = tokenizer.encode('Once upon').map((id) => ({ id, fromPrompt: true }))
  await generate({
    generator: model.generator,
    tokenizer,
    prompt: 'Once upon',
    settings: { ...DEFAULT_SETTINGS, maxTokens: 5 },
    blockSize: 64,
    onToken: (t) => tokens.push(t),
  })
  return tokens
}

describe('output views', () => {
  it('text view shows the prompt and the generated text', async () => {
    const model = mockModel()
    const tokens = await someTokens(model)
    render(<TextView model={model} tokens={tokens} running={false} />)
    expect(screen.getByText('Once upon')).toBeInTheDocument()
  })

  it('token view shows one chip per token with its id', async () => {
    const model = mockModel()
    const tokens = await someTokens(model)
    render(<TokenView model={model} tokens={tokens} running={false} />)
    expect(screen.getAllByRole('listitem')).toHaveLength(tokens.length)
    expect(screen.getAllByText(String(tokens[0].id)).length).toBeGreaterThan(0)
  })

  it('probability view lists the top 10 guesses for the picked token', async () => {
    const model = mockModel()
    const tokens = await someTokens(model)
    render(<ProbabilityView model={model} tokens={tokens} running={false} />)
    // the first generated token is picked to begin with
    expect(screen.getByText('Top guesses for token 1')).toBeInTheDocument()
    expect(screen.getAllByRole('listitem')).toHaveLength(10)
    // picking another token switches the list
    fireEvent.click(screen.getAllByRole('button')[2])
    expect(screen.getByText('Top guesses for token 3')).toBeInTheDocument()
  })

  it('probability view asks for text when there is none', () => {
    render(<ProbabilityView model={mockModel()} tokens={[]} running={false} />)
    expect(screen.getByText(/Generate some text first/)).toBeInTheDocument()
  })

  it('attention view shows layer and head controls once the maps arrive', async () => {
    const model = mockModel()
    const tokens = await someTokens(model)
    render(<AttentionView model={model} tokens={tokens} running={false} />)
    await waitFor(() => expect(screen.getByRole('group', { name: 'Layer' })).toBeInTheDocument())
    expect(screen.getByRole('group', { name: 'Head' })).toBeInTheDocument()
  })

  it('attention view explains itself when the model has no attention', async () => {
    const model = mockModel(false)
    const tokens = await someTokens(model)
    render(<AttentionView model={model} tokens={tokens} running={false} />)
    expect(screen.getByText(/exported without attention maps/)).toBeInTheDocument()
  })
})
