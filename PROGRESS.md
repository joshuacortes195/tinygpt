# Progress

Last updated: 2026-10-08

| Phase | What | Status |
| --- | --- | --- |
| 0 | Setup and scaffolding | done |
| 1 | Byte-level BPE tokenizer (Python + TypeScript) | done |
| 2 | GPT model and training loop | done |
| 3 | ONNX export and parity | done |
| 4 | Web app core | done |
| 5 | "Look inside the model" visualizations | done |
| 6 | "How I built it" page and deployment | done |
| 7 | Real training | done |
| 8 | Fine-tuned comparison model (stretch) | done |

## Decisions

- **Dev machine is the Windows desktop** (Ryzen 7 3700X, 16 GB RAM, RTX 3060 12 GB), not the Intel Mac the plan was first written for. So the venv uses Python 3.12 and a current CUDA build of PyTorch (2.6.0+cu124). The code still picks `cuda` / `mps` / `cpu` on its own, and the Intel Mac pins (`torch==2.2.2`, `numpy<2`, Python 3.11) are in the README.
- **Hosting is GitHub Pages** instead of Netlify. The site is a plain static build with relative paths, so it can move to Netlify later without code changes.
- **Deploy is a `gh-pages` branch push**, not an Actions workflow. Files over 100 MB can't be pushed to GitHub, so big ONNX files are split into parts that the site stitches back together when it downloads them.
- **Real training was started right after Phase 2** instead of last, so the GPU works while the website gets built. The site still only depends on the manifest, never on a specific model.
- **Step counts are sized to about 4 hours of GPU time**: small 25,000 steps (~50 min at 134k tokens/sec), base 36,000 steps (~2h45m at 60k tokens/sec).

## Phase notes

**Phase 0.** Repo layout, `.gitignore`, README, web scaffold (Vite + React + TypeScript + Tailwind, Vitest). torch imports and runs a matmul on the GPU.

**Phase 1.** BPE tokenizer in Python and TypeScript, 4,096 vocab. Full TinyStories V2 tokenized: 558,888,207 train tokens, 5,642,898 val tokens. 20 golden cases pass in both languages.

**Phase 2.** GPT model, trainer, three configs, sampling script. v0 smoke model: 735k params, 6,000 steps, val loss 2.38, 74 seconds on the GPU.

**Phase 3.** ONNX export (fp32 + int8) with parity checks. v0 exported to `web/public/models/v0-smoke/` and listed in `manifest.json`. fp32 max logit diff 1.3e-5, int8 top-1 agreement 100%.

**Phase 4.** Playground with model picker, streaming output, sliders, seed, Stop. `TextGenerator` interface with a mock and a Web Worker ONNX implementation (WebGPU first, WebAssembly fallback). Tested in headless Chrome at desktop and phone sizes, light and dark, on both backends.

**Phase 5.** Token, probability and attention views inside the playground's output panel, plus a Compare page. All work with v0, and the attention view degrades when a model has no attention output.

**Phase 6.** Build page (stats, loss curve and architecture all read from the model's files), `TRAINING.md`, README. Live at https://joshuacortes195.github.io/tinygpt/ running v0. Deploy with `npm run deploy` from `web/`. Verified against the live URL in headless Chrome.

**Phase 7.** small: 25,000 steps, 53 minutes, val loss 1.342. base: 36,000 steps, 2 h 47 min, val loss 1.208. Both exported (fp32 + int8) and live, base is the default. In Chrome on the desktop, base does 76 tokens/sec on WebGPU, 13 on WebAssembly int8 and 9 on WebAssembly fp32. Full numbers are in LEARNING.md.

**Phase 8.** SmolLM2-135M fine-tuned with my own LoRA code (rank 8, 0.34% of the weights, 3,000 steps, 24 minutes). Val loss went from 2.131 to 1.666 per token. Per byte that is 0.405 against 0.303 for base, so the from-scratch model still wins on this data. Exported to ONNX (165 MB, int8) and offered as an optional download on the Compare page, where it runs at about 19 tokens/sec on the CPU.

## How to retrain or add a model

Follow TRAINING.md. A run is finished when the last line of `model/runs/<name>/metrics.jsonl` has `"step"` equal to `max_steps` and a `val_loss`. If a run died part way, restart it with `--resume`. Don't run two GPU jobs at once: together they fill the 12 GB of GPU memory and both crawl.
## Open items

- **CI workflow is not on GitHub yet.** The saved GitHub login doesn't have the `workflow` permission, so pushing `.github/workflows/ci.yml` is rejected. The file is committed on the local `ci` branch. To turn CI on: run `gh auth refresh -s workflow`, then `git checkout main && git merge ci && git push`.
- Safari can't be tested from this Windows machine. Chrome and Edge are covered; Safari needs a check on a Mac or iPhone.
