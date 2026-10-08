# TinyGPT

A GPT-style language model I built from scratch: my own tokenizer, my own transformer, my own training loop. It runs in your browser, so there is no server and nothing to pay for.

Work in progress. See [PROGRESS.md](PROGRESS.md) for where things stand and [LEARNING.md](LEARNING.md) for notes on how each part works.

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
```

## Setup

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

No NVIDIA GPU? Install the CPU build instead: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

On an Intel Mac, PyTorch stopped shipping wheels after 2.2, so use Python 3.11 with `pip install torch==2.2.2 "numpy<2"`.
