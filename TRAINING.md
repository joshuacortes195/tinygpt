# Training runbook

How to train the real models on a Windows desktop with an NVIDIA GPU, export them, and put them on the site. Every command runs from the repo root unless it says otherwise.

Numbers below are from my machine: RTX 3060 12 GB, Ryzen 7 3700X, 16 GB RAM.

## 1. Setup (once)

```powershell
# python 3.12 and a virtual environment
py -3.12 -m venv .venv
.venv\Scripts\activate

# pytorch built for CUDA 12.4, then the project itself
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".\model[dev]"

# check the GPU is visible, this should print True and the card's name
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `False`, update the NVIDIA driver and reinstall torch with the command above. `nvidia-smi` should list the card.

## 2. Prepare the data (once, about 10 minutes)

```powershell
cd model
python scripts\prepare_data.py
```

This downloads TinyStories (2.2 GB), trains the tokenizer on the first 50 MB, and tokenizes everything into `model/data/train.bin` (1.1 GB) and `val.bin`. Each step is skipped if its output already exists. Expect about 559M training tokens.

Useful flags: `--vocab-size 4096`, `--tokenizer-sample-mb 50`, `--max-train-mb 100` (only tokenize a slice, for a quick test).

## 3. Train

```powershell
cd model
python scripts\train.py configs\small.yaml
python scripts\train.py configs\base.yaml
```

| Config | Params | Steps | Speed | Time | VRAM |
| --- | --- | --- | --- | --- | --- |
| small | 10.5M | 25,000 | ~134k tokens/sec | ~50 min | about 3 GB |
| base | 27.4M | 36,000 | ~60k tokens/sec | ~2h 45m | about 4 GB |

Each step is 64 sequences of 256 tokens. To train longer, raise `max_steps` in the YAML. One pass over the dataset is about 34,000 steps.

### Watching the loss

The script prints a line every 100 steps and a validation loss every 1,000. The same numbers go to `model/runs/<name>/metrics.jsonl`.

```powershell
# follow a run from another terminal
Get-Content model\runs\base\metrics.jsonl -Wait -Tail 5
```

What healthy looks like: loss starts near 8.3 (random guessing over 4,096 tokens), drops under 3 within the first thousand steps, and then creeps down slowly. Validation loss should track training loss closely. If validation starts going up while training keeps falling, the model is overfitting and it is time to stop.

### Resuming after an interruption

A checkpoint is saved every 1,000 steps to `model/runs/<name>/ckpt.pt`. If the run stops for any reason:

```powershell
python scripts\train.py configs\base.yaml --resume
```

It picks up at the last saved step with the same optimizer state and the same random number streams.

### If you run out of GPU memory

Halve `batch_size` and double `grad_accum` in the YAML. The effective batch stays the same.

### Reading a sample

```powershell
python scripts\sample.py runs\base\ckpt.pt --prompt "Once upon a time" --max-tokens 200
```

## 4. Backup plan: Kaggle

If the desktop is not available, a free Kaggle notebook with a GPU works. Turn on **GPU T4** and **Internet** in the notebook settings, then:

```python
!git clone https://github.com/joshuacortes195/tinygpt.git
%cd tinygpt
!pip install -e "./model[dev]"

%cd model
!python scripts/prepare_data.py

# write checkpoints to /kaggle/working so they survive the session
!sed -i 's#out_dir: runs/base#out_dir: /kaggle/working/base#' configs/base.yaml
!python scripts/train.py configs/base.yaml
```

Kaggle sessions stop after about 12 hours. If one ends early, start a new session, put the old `ckpt.pt` and `metrics.jsonl` back in `/kaggle/working/base/`, and run the same command with `--resume`.

When it finishes, download `ckpt.pt` and `metrics.jsonl` from the notebook's output panel and put them in `model/runs/base/` on your own machine. You also need `model/data/tokenizer.json` and `model/data/val.bin` from the same run for the export step, so download those too.

## 5. Export and publish

```powershell
cd model
python scripts\export_onnx.py runs\small\ckpt.pt --id small --name "Small (10.5M)" --description "First real run. About 50 minutes on one GPU."
python scripts\export_onnx.py runs\base\ckpt.pt --id base --name "Base (27M)" --description "The full model. About 3 hours on one GPU." --default
```

Each command:

- writes `web/public/models/<id>/` with the fp32 and int8 ONNX files, the tokenizer, `config.json` and `metrics.json`,
- checks the ONNX output against PyTorch and fails loudly if they disagree,
- splits any file over 45 MB into parts (GitHub rejects files over 100 MB),
- adds the model to `web/public/models/manifest.json`. `--default` makes it the model the site opens with.

Then publish:

```powershell
cd web
npm run deploy
```

That builds the site and pushes it to the `gh-pages` branch. GitHub Pages picks it up within a minute or two. No code changes are needed: the site reads everything from the manifest.

To remove a model, delete its folder and its entry in `manifest.json`.

## Checklist

- [ ] `nvidia-smi` shows the GPU and torch reports `cuda` available
- [ ] `model/data/train.bin`, `val.bin` and `tokenizer.json` exist
- [ ] `small` finished: last line of `runs/small/metrics.jsonl` has `"step": 25000` and a `val_loss`
- [ ] `base` finished: last line of `runs/base/metrics.jsonl` has `"step": 36000` and a `val_loss`
- [ ] both exported with fp32 parity under 0.001
- [ ] `manifest.json` lists them and `base` is the default
- [ ] `npm run deploy`, then load the live site and generate a story
