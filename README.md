# TinyGPT

A GPT-style language model I built from scratch: my own tokenizer, my own transformer, my own training loop. It runs entirely in your browser, so there is no server and nothing to pay for.

**Try it: https://joshuacortes195.github.io/tinygpt/**

<!-- TODO: record a short GIF of the playground and save it as docs/demo.gif -->
![Demo of the playground](docs/demo.gif)

## What it does

- **Playground.** Type a prompt and watch the model write a story one token at a time. Sliders for temperature, top-k, top-p and length, plus a seed so runs are repeatable.
- **Look inside.** See how your text was split into tokens, what the model's top 10 guesses were at every step, and which earlier words each attention head was looking at.
- **Compare.** Run the same prompt through two models side by side.
- **How I built it.** Architecture, model stats and the real loss curves from training.

## How it works

```
text ──> BPE tokenizer ──> token + position embeddings
                                    │
                    ┌───────────────▼────────────────┐
                    │ LayerNorm → causal attention ─+ │
                    │ LayerNorm → MLP (GELU) ───────+ │  x N blocks
                    └───────────────┬────────────────┘
                                    ▼
                  LayerNorm → output layer (tied to embeddings)
                                    ▼
                        probabilities for the next token
```

- **Tokenizer:** byte-level BPE with a 4,096 token vocab, written in Python and ported to TypeScript. Both are tested against the same golden cases so they always produce identical ids.
- **Model:** decoder-only transformer with hand-written multi-head causal self-attention, pre-LayerNorm blocks, GELU MLPs and tied input/output embeddings. No `nn.Transformer`, no `nn.MultiheadAttention`, no Hugging Face.
- **Training:** AdamW, cosine schedule with warmup, gradient clipping, mixed precision on CUDA, and checkpoints that resume bit-for-bit.
- **Data:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), about 559M tokens of short children's stories.
- **Browser:** the model is exported to ONNX and run with ONNX Runtime Web in a Web Worker. WebGPU when the browser has it, WebAssembly otherwise.

| Model | Parameters | Layers / heads / width | Context |
| --- | --- | --- | --- |
| v0 smoke | 0.7M | 3 / 3 / 96 | 64 |
| small | 10.5M | 5 / 6 / 384 | 256 |
| base | 27.4M | 8 / 8 / 512 | 256 |

More detail on every part, with the reasoning behind it, is in [LEARNING.md](LEARNING.md).

## Layout

```
model/            Python: tokenizer, GPT, training, ONNX export
  tinygpt/        the package
  configs/        smoke.yaml, small.yaml, base.yaml
  scripts/        prepare_data, train, sample, export_onnx
  tests/
web/              React + TypeScript + Vite + Tailwind
  public/models/  manifest.json + one folder per model
shared/fixtures/  tokenizer test cases checked by both Python and TypeScript
LEARNING.md       notes on how each part works
TRAINING.md       step by step guide for training and publishing a model
PROGRESS.md       what is done and what is left
```

## Run it yourself

```bash
# python side
python -m venv .venv
.venv\Scripts\activate            # mac/linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # nvidia gpu
pip install -e "./model[dev]"
cd model && pytest

# web side
cd web
npm install
npm test
npm run dev
```

No NVIDIA GPU? Install the CPU build instead: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. The smoke config trains on a CPU in a few minutes.

On an Intel Mac, PyTorch stopped shipping wheels after 2.2, so use Python 3.11 with `pip install torch==2.2.2 "numpy<2"`.

To train the real models and publish them, follow [TRAINING.md](TRAINING.md).

## Adding a model to the site

The site never hardcodes a model. It reads `web/public/models/manifest.json`, so adding one is:

```bash
cd model
python scripts/export_onnx.py runs/base/ckpt.pt --id base --name "Base (27M)" --default
cd ../web
npm run deploy
```
