// 27448320 -> "27.4M"
export function formatCount(n: number | null | undefined): string {
  if (n === null || n === undefined) return 'n/a'
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`
  return String(n)
}

// 4559049 -> "4.6 MB"
export function formatBytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return 'n/a'
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`
  return `${Math.max(1, Math.round(n / 1e3))} KB`
}

// 9930 seconds -> "2h 45m"
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return 'n/a'
  if (seconds < 90) return `${Math.round(seconds)}s`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m`
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`
}

// 0.3412 -> "34.1%"
export function formatPercent(p: number): string {
  if (p >= 0.0995) return `${(p * 100).toFixed(1)}%`
  if (p >= 0.001) return `${(p * 100).toFixed(2)}%`
  return '<0.1%'
}

// makes spaces and newlines visible when a token is shown on its own
export function visibleToken(text: string): string {
  if (text === '') return '∅'
  return text.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '→').replace(/\r/g, '␍')
}
