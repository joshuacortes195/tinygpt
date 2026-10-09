import { useEffect, useState } from 'react'
import { Header, type Page } from './components/Header'
import { REPO_URL } from './lib/links'
import { loadManifest } from './lib/manifest'
import { MOCK_ENTRY } from './lib/models'
import type { ExtraEntry, ManifestEntry } from './lib/types'
import { Build } from './pages/Build'
import { Compare } from './pages/Compare'
import { TryIt } from './pages/TryIt'

// which page the url hash points at
function pageFromHash(): Page {
  const hash = window.location.hash.replace(/^#\/?/, '')
  if (hash.startsWith('compare')) return 'compare'
  if (hash.startsWith('build')) return 'build'
  return 'try'
}

type ManifestState =
  | { status: 'loading' }
  | { status: 'ready'; models: ManifestEntry[]; defaultId: string; extras: ExtraEntry[]; compare: string[] }
  | { status: 'error'; message: string }

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash)
  // pages are created the first time you open them and then kept, so nothing is lost when you switch
  const [visited, setVisited] = useState<Set<Page>>(() => new Set([pageFromHash()]))
  const [manifest, setManifest] = useState<ManifestState>({ status: 'loading' })
  const [selectedId, setSelectedId] = useState('')

  // keeps the page in sync with the back and forward buttons
  useEffect(() => {
    const onHashChange = () => {
      const next = pageFromHash()
      setPage(next)
      setVisited((prev) => new Set(prev).add(next))
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  // the manifest decides which models exist, the app has no model names of its own
  useEffect(() => {
    const wantMock = new URLSearchParams(window.location.search).has('mock')
    loadManifest()
      .then((m) => {
        const models = wantMock ? [...m.models, MOCK_ENTRY] : m.models
        const defaultId = models.some((x) => x.id === m.default) ? m.default : (models[0]?.id ?? '')
        setManifest({ status: 'ready', models, defaultId, extras: m.extras ?? [], compare: m.compare ?? [] })
        setSelectedId(wantMock ? MOCK_ENTRY.id : defaultId)
      })
      .catch((err: unknown) => {
        // with ?mock=1 the site still works with no model files at all
        if (wantMock) {
          setManifest({ status: 'ready', models: [MOCK_ENTRY], defaultId: MOCK_ENTRY.id, extras: [], compare: [] })
          setSelectedId(MOCK_ENTRY.id)
        } else {
          setManifest({ status: 'error', message: err instanceof Error ? err.message : String(err) })
        }
      })
  }, [])

  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-30 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <Header page={page} />

      <main id="main" className="flex-1">
        {manifest.status === 'loading' && (
          <div className="frame cell grid gap-4 sm:py-12" aria-label="Loading">
            <div className="skeleton h-10 w-2/3 max-w-lg" />
            <div className="skeleton h-5 w-full max-w-xl" />
            <div className="skeleton mt-6 h-48 w-full" />
          </div>
        )}

        {manifest.status === 'error' && (
          <div className="frame cell sm:py-16" role="alert">
            <h1 className="text-2xl font-semibold tracking-tight">The model list did not load.</h1>
            <p className="mt-3 max-w-[60ch] leading-relaxed text-muted">
              {manifest.message}. Check your connection and reload the page.
            </p>
            <button type="button" className="btn btn-primary mt-6" onClick={() => window.location.reload()}>
              Reload
            </button>
          </div>
        )}

        {manifest.status === 'ready' && manifest.models.length === 0 && (
          <div className="frame cell sm:py-16">
            <h1 className="text-2xl font-semibold tracking-tight">No models are listed yet.</h1>
            <p className="mt-3 leading-relaxed text-muted">
              Add a model folder and an entry in manifest.json and it will show up here.
            </p>
          </div>
        )}

        {manifest.status === 'ready' && manifest.models.length > 0 && (
          <>
            <div hidden={page !== 'try'}>
              {visited.has('try') && (
                <TryIt models={manifest.models} selectedId={selectedId} onSelect={setSelectedId} />
              )}
            </div>
            <div hidden={page !== 'compare'}>
              {visited.has('compare') && <Compare models={manifest.models} defaultId={manifest.defaultId} extras={manifest.extras} pair={manifest.compare} />}
            </div>
            <div hidden={page !== 'build'}>
              {visited.has('build') && <Build models={manifest.models} defaultId={manifest.defaultId} />}
            </div>
          </>
        )}
      </main>

      <footer className="border-t border-border">
        <div className="frame num flex flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 py-5 text-xs text-muted sm:px-7">
          <p>Built by Joshua Cortes. The model runs on your device and nothing you type is uploaded.</p>
          <a href={REPO_URL} className="flex min-h-11 items-center text-text hover:text-accent">
            Source on GitHub
          </a>
        </div>
      </footer>
    </div>
  )
}
