import { GithubLogoIcon, MoonIcon, SunIcon } from '@phosphor-icons/react'
import { useEffect, useState } from 'react'
import { REPO_URL } from '../lib/links'

export type Page = 'try' | 'compare' | 'build'

const LINKS: { page: Page; label: string; href: string }[] = [
  { page: 'try', label: 'Try it', href: '#/' },
  { page: 'compare', label: 'Compare', href: '#/compare' },
  { page: 'build', label: 'How I built it', href: '#/build' },
]

// reads the theme the page started with
function currentTheme(): 'light' | 'dark' {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'
}

export function Header({ page }: { page: Page }) {
  const [theme, setTheme] = useState<'light' | 'dark'>(currentTheme)

  // applies the theme and remembers the choice for next visit
  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    try {
      localStorage.setItem('theme', next)
    } catch {
      // private mode can block storage, the toggle still works for this visit
    }
  }

  return (
    <header className="sticky top-0 z-20 border-b border-border bg-bg/90 backdrop-blur">
      <div className="frame flex h-14 items-center gap-2 px-4 sm:gap-6 sm:px-7">
        {/* site name with a little cursor block next to it */}
        <a href="#/" className="num flex shrink-0 items-center gap-2 text-[0.9375rem] font-semibold tracking-tight">
          <span className="h-4 w-2 bg-accent" aria-hidden="true" />
          tinygpt
        </a>

        {/* page links, they scroll sideways on very small phones */}
        <nav aria-label="Pages" className="-mx-1 flex min-w-0 flex-1 gap-1 overflow-x-auto px-1">
          {LINKS.map((link) => (
            <a
              key={link.page}
              href={link.href}
              aria-current={page === link.page ? 'page' : undefined}
              className={`num flex h-14 shrink-0 items-center border-b-2 px-2.5 text-[0.8125rem] transition-colors sm:px-3 ${
                page === link.page ? 'border-accent text-accent' : 'border-transparent text-muted hover:text-text'
              }`}
            >
              {link.label}
            </a>
          ))}
        </nav>

        <a href={REPO_URL} className="btn btn-ghost w-11 shrink-0 px-0" aria-label="Source code on GitHub">
          <GithubLogoIcon size={20} />
        </a>
        <button
          type="button"
          onClick={toggleTheme}
          className="btn btn-ghost w-11 shrink-0 px-0"
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
        >
          {theme === 'dark' ? <SunIcon size={20} /> : <MoonIcon size={20} />}
        </button>
      </div>
    </header>
  )
}
