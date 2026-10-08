export const REPO_URL = 'https://github.com/joshuacortes195/tinygpt'

// link to a heading inside LEARNING.md on github
export function learningLink(anchor: string): string {
  return `${REPO_URL}/blob/main/LEARNING.md#${anchor}`
}
