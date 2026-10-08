# Progress

Last updated: 2026-10-08

| Phase | What | Status |
| --- | --- | --- |
| 0 | Setup and scaffolding | done |
| 1 | Byte-level BPE tokenizer (Python + TypeScript) | not started |
| 2 | GPT model and training loop | not started |
| 3 | ONNX export and parity | not started |
| 4 | Web app core | not started |
| 5 | "Look inside the model" visualizations | not started |
| 6 | "How I built it" page and deployment | not started |
| 7 | Real training | not started |
| 8 | Fine-tuned comparison model (stretch) | not started |

## Decisions

- **Dev machine is the Windows desktop** (Ryzen 7 3700X, 16 GB RAM, RTX 3060 12 GB), not the Intel Mac the plan was first written for. So the venv uses Python 3.12 and a current CUDA build of PyTorch (2.6.0+cu124). The code still picks `cuda` / `mps` / `cpu` on its own, and the Intel Mac pins (`torch==2.2.2`, `numpy<2`, Python 3.11) are in the README.
- **Hosting is GitHub Pages** instead of Netlify. The site is a plain static build with relative paths, so it can move to Netlify later without code changes.
- **Big model files do not go in git.** They are attached to a GitHub Release and the deploy workflow pulls them into the site build. The tiny v0 model is small enough to commit.

## Phase 0 notes

- Repo layout, `.gitignore`, README, CI workflow (Python tests on CPU, web tests + build).
- `web/` scaffolded with Vite + React + TypeScript + Tailwind, Vitest for tests.
- Checked the environment: torch imports and runs a matmul on the GPU.

## Open items

- Safari can't be tested from this Windows machine. Chrome and Edge are covered; Safari needs a check on a Mac or iPhone.
