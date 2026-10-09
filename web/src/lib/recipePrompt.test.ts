import { describe, expect, it } from 'vitest'
import { dishTitle, editDistance, fixWord, preparePrompt } from './recipePrompt'
import type { ManifestEntry } from './types'

const recipe: ManifestEntry = { id: 'r', name: 'r', description: '', path: 'r', format: 'recipe' }
const plain: ManifestEntry = { id: 'p', name: 'p', description: '', path: 'p' }

describe('recipe prompts', () => {
  it('counts edits between two words', () => {
    expect(editDistance('pizza', 'pizza', 2)).toBe(0)
    expect(editDistance('piza', 'pizza', 2)).toBe(1)
    // swapped letters are one mistake
    expect(editDistance('chikcen', 'chicken', 2)).toBe(1)
  })

  it('fixes common typos', () => {
    expect(fixWord('piza')).toBe('pizza')
    expect(fixWord('chiken')).toBe('chicken')
    expect(fixWord('noodels')).toBe('noodles')
    expect(fixWord('spagetti')).toBe('spaghetti')
  })

  it('leaves real words and short words alone', () => {
    expect(fixWord('ramen')).toBe('ramen')
    expect(fixWord('pie')).toBe('pie')
    expect(fixWord('zzzzqqqq')).toBe('zzzzqqqq')
  })

  it('writes the title the way the training data does', () => {
    expect(dishTitle('  ramen   noodels ')).toBe('Ramen Noodles')
  })

  it('adds the ingredients heading for recipe models only', () => {
    expect(preparePrompt('piza', recipe)).toBe('Pizza\n\nIngredients:\n')
    expect(preparePrompt('piza', plain)).toBe('piza')
    expect(preparePrompt('', recipe)).toBe('')
    // a prompt that already has its own lines is not touched
    expect(preparePrompt('Pizza\n\nIngredients:\n- 1 c. flour', recipe)).toBe('Pizza\n\nIngredients:\n- 1 c. flour')
  })
})
