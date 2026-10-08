/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  // relative paths so the same build works on github pages, netlify or any subfolder
  base: './',
  plugins: [react(), tailwindcss()],
  worker: { format: 'es' },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    passWithNoTests: true,
  },
})
