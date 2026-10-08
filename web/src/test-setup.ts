import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// clear the page between tests so one test can't see another's output
afterEach(() => cleanup())
