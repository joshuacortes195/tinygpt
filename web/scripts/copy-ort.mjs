// copies the onnx runtime wasm files into public/ so the site serves them itself
import { cpSync, mkdirSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const from = join(root, 'node_modules', 'onnxruntime-web', 'dist')
const to = join(root, 'public', 'ort')

mkdirSync(to, { recursive: true })
// only the runtime pieces the browser fetches, not the whole package
const files = readdirSync(from).filter((f) => /^ort-wasm-simd-threaded.*\.(wasm|mjs)$/.test(f))
for (const f of files) cpSync(join(from, f), join(to, f))
console.log(`copied ${files.length} onnx runtime files`)
