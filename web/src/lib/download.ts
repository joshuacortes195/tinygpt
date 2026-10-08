// downloads one or more files into a single buffer and reports progress as it goes
export async function fetchBytes(
  urls: string[],
  totalBytes: number,
  onProgress?: (loaded: number, total: number) => void,
): Promise<Uint8Array> {
  const chunks: Uint8Array[] = []
  let loaded = 0

  for (const url of urls) {
    const res = await fetch(url)
    if (!res.ok || !res.body) throw new Error(`Download failed for ${url} (${res.status})`)
    // read the response piece by piece so the progress bar can move
    const reader = res.body.getReader()
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      loaded += value.length
      onProgress?.(loaded, Math.max(totalBytes, loaded))
    }
  }

  // glue all the pieces together
  const out = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.length
  }
  return out
}
