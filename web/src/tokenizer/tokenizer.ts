// shape of the tokenizer.json file the python side saves
export interface TokenizerData {
  version: number
  pattern: string
  merges: [number, number][]
  special_tokens: Record<string, number>
}

const encoder = new TextEncoder()
const decoder = new TextDecoder('utf-8')

// same byte-level bpe as model/tinygpt/tokenizer.py, they must always agree
export class BPETokenizer {
  readonly merges: [number, number][]
  readonly specialTokens: Record<string, number>
  readonly vocabSize: number
  private regex: RegExp
  private specialRegex: RegExp | null
  private vocab: Uint8Array[] = []
  private ranks = new Map<number, number>()
  private specialIds = new Map<number, string>()
  private cache = new Map<string, number[]>()

  constructor(data: TokenizerData) {
    this.merges = data.merges
    this.specialTokens = data.special_tokens
    // the pattern comes from the json so both languages split text the same way
    this.regex = new RegExp(data.pattern, 'gu')

    // ids 0-255 are the raw bytes
    for (let i = 0; i < 256; i++) this.vocab.push(new Uint8Array([i]))
    // every merge makes one new token out of two older ones
    data.merges.forEach(([a, b], i) => {
      const joined = new Uint8Array(this.vocab[a].length + this.vocab[b].length)
      joined.set(this.vocab[a], 0)
      joined.set(this.vocab[b], this.vocab[a].length)
      this.vocab.push(joined)
      this.ranks.set(pairKey(a, b), 256 + i)
    })

    // splits text around special tokens, longest first so none get cut short
    const names = Object.keys(data.special_tokens).sort((x, y) => y.length - x.length)
    for (const name of names) this.specialIds.set(data.special_tokens[name], name)
    this.specialRegex = names.length ? new RegExp(`(${names.map(escapeRegex).join('|')})`, 'u') : null
    this.vocabSize = 256 + data.merges.length + names.length
  }

  // turns one chunk into token ids by replaying the merges in the order they were learned
  private encodeChunk(chunk: string): number[] {
    const cached = this.cache.get(chunk)
    if (cached) return cached
    let ids = Array.from(encoder.encode(chunk))
    while (ids.length >= 2) {
      // find the pair that was learned earliest
      let bestRank = Infinity
      let bestA = -1
      let bestB = -1
      for (let i = 0; i < ids.length - 1; i++) {
        const rank = this.ranks.get(pairKey(ids[i], ids[i + 1]))
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank
          bestA = ids[i]
          bestB = ids[i + 1]
        }
      }
      if (bestRank === Infinity) break
      // glue that pair together everywhere in the chunk
      const merged: number[] = []
      let j = 0
      while (j < ids.length) {
        if (j < ids.length - 1 && ids[j] === bestA && ids[j + 1] === bestB) {
          merged.push(bestRank)
          j += 2
        } else {
          merged.push(ids[j])
          j += 1
        }
      }
      ids = merged
    }
    // keeps the cache from growing forever
    if (this.cache.size < 50000) this.cache.set(chunk, ids)
    return ids
  }

  // text to token ids
  encode(text: string, allowSpecial = true): number[] {
    const parts = allowSpecial && this.specialRegex ? text.split(this.specialRegex) : [text]
    const ids: number[] = []
    for (const part of parts) {
      if (!part) continue
      // special tokens map straight to their id
      if (allowSpecial && Object.hasOwn(this.specialTokens, part)) {
        ids.push(this.specialTokens[part])
        continue
      }
      for (const match of part.matchAll(this.regex)) {
        for (const id of this.encodeChunk(match[0])) ids.push(id)
      }
    }
    return ids
  }

  // the raw bytes behind one token
  tokenBytes(id: number): Uint8Array {
    const special = this.specialIds.get(id)
    if (special !== undefined) return encoder.encode(special)
    return this.vocab[id] ?? new Uint8Array()
  }

  // token ids back to text
  decode(ids: number[]): string {
    const pieces = ids.map((id) => this.tokenBytes(id))
    const bytes = new Uint8Array(pieces.reduce((n, p) => n + p.length, 0))
    let offset = 0
    for (const p of pieces) {
      bytes.set(p, offset)
      offset += p.length
    }
    // half finished characters become the replacement symbol instead of crashing
    return decoder.decode(bytes)
  }

  // text of a single token for display, with a marker when it is only part of a character
  tokenLabel(id: number): string {
    const text = new TextDecoder('utf-8', { fatal: false }).decode(this.tokenBytes(id))
    return text
  }

  isSpecial(id: number): boolean {
    return this.specialIds.has(id)
  }
}

// packs a pair of ids into one number so it can be a map key
function pairKey(a: number, b: number): number {
  return a * 65536 + b
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
