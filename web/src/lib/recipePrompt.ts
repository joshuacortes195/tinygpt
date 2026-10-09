import words from './dishWords.json'
import { DEFAULT_SETTINGS, type ManifestEntry, type SamplingSettings } from './types'

// the most common words in recipe titles, most common first
const DISH_WORDS: string[] = words
const KNOWN = new Set(DISH_WORDS)

// how many single letter edits it takes to turn one word into the other, giving up past the limit
export function editDistance(a: string, b: string, limit: number): number {
  if (Math.abs(a.length - b.length) > limit) return limit + 1
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i)
  // the row before the previous one, needed to spot swapped letters
  let prevPrev: number[] = []
  for (let i = 1; i <= a.length; i++) {
    const row = [i]
    let best = i
    for (let j = 1; j <= b.length; j++) {
      const swap = prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, swap)
      // two letters typed in the wrong order count as one mistake
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        row[j] = Math.min(row[j], prevPrev[j - 2] + 1)
      }
      best = Math.min(best, row[j])
    }
    if (best > limit) return limit + 1
    prevPrev = prev
    prev = row
  }
  return prev[b.length]
}

// swaps a misspelled word for the closest word that shows up in real recipe titles
export function fixWord(word: string): string {
  const lower = word.toLowerCase()
  // short words, numbers and words we already know are left alone
  if (lower.length < 4 || !/^[a-z]+$/.test(lower) || KNOWN.has(lower)) return word
  // longer words are allowed one more mistake
  const limit = lower.length >= 7 ? 2 : 1
  let best = word
  let bestDistance = limit + 1
  // the list is sorted by how common a word is, so the first match at a distance is the likeliest
  for (const candidate of DISH_WORDS) {
    const distance = editDistance(lower, candidate, limit)
    if (distance < bestDistance) {
      best = candidate
      bestDistance = distance
      if (distance === 1) break
    }
  }
  return best
}

// turns whatever was typed into the title of a recipe, the way titles looked in the training data
export function dishTitle(text: string): string {
  return text
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => {
      const fixed = fixWord(word)
      // titles in the training data start every word with a capital
      return fixed.charAt(0).toUpperCase() + fixed.slice(1)
    })
    .join(' ')
}

// gets a prompt ready for the model that is about to read it
export function preparePrompt(prompt: string, entry: ManifestEntry | null | undefined): string {
  // only recipe models need this, and a prompt that already has its own layout is left alone
  if (entry?.format !== 'recipe' || prompt.includes('\n')) return prompt
  const title = dishTitle(prompt)
  // an empty prompt lets the model pick the dish itself
  if (!title) return ''
  // the heading tells the model the title is finished and the list starts here
  return `${title}\n\nIngredients:\n`
}

// the sampling settings a model starts with
export function startSettings(entry: ManifestEntry | null | undefined): SamplingSettings {
  // recipes stay on topic better with less randomness, and need more room to finish
  if (entry?.format === 'recipe') return { ...DEFAULT_SETTINGS, temperature: 0.5, maxTokens: 200 }
  return DEFAULT_SETTINGS
}
