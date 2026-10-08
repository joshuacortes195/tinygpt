import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .model import GPT, GPTConfig

# the model/ folder, so paths in configs work no matter where you run from
MODEL_ROOT = Path(__file__).resolve().parent.parent


# reads a yaml config and fills in the defaults
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    defaults = {
        "batch_size": 32,
        "grad_accum": 1,
        "max_steps": 1000,
        "lr": 1e-3,
        "min_lr": 1e-4,
        "warmup_steps": 100,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "eval_interval": 250,
        "eval_batches": 50,
        "log_interval": 50,
        "seed": 1337,
        "compile": False,
        "device": "auto",
    }
    cfg["train"] = {**defaults, **cfg.get("train", {})}
    return cfg


# turns a path from the config into a real path
def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else MODEL_ROOT / p


# cuda on the desktop, mps on newer macs, cpu everywhere else
def pick_device(name="auto"):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# warm up in a straight line, then follow a cosine curve down to min_lr
def get_lr(step, t):
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    if step >= t["max_steps"]:
        return t["min_lr"]
    progress = (step - t["warmup_steps"]) / max(1, t["max_steps"] - t["warmup_steps"])
    return t["min_lr"] + 0.5 * (1.0 + math.cos(math.pi * progress)) * (t["lr"] - t["min_lr"])


# grabs random windows of tokens, y is x shifted one token ahead
def get_batch(data, batch_size, block_size, generator, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator).tolist()
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix]).astype(np.int64)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


# average loss over a fixed set of validation batches, so numbers are comparable between evals
@torch.no_grad()
def estimate_loss(model, data, t, block_size, device, use_amp):
    model.eval()
    g = torch.Generator().manual_seed(t["seed"] + 1)
    losses = []
    for _ in range(t["eval_batches"]):
        x, y = get_batch(data, t["batch_size"], block_size, g, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# saves everything needed to pick the run back up later
def save_checkpoint(path, model, optimizer, scaler, step, batch_gen, cfg, best_val, tokens_seen, elapsed):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_batch": batch_gen.get_state(),
        "config": cfg,
        "best_val": best_val,
        "tokens_seen": tokens_seen,
        "elapsed": elapsed,
    }
    # write to a temp file first so a crash mid-save can't wreck the old checkpoint
    tmp = str(path) + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def train(cfg, resume=False, stop_at=None, quiet=False):
    t = cfg["train"]
    device = pick_device(t["device"])
    use_amp = device == "cuda"
    out_dir = resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    metrics_path = out_dir / "metrics.jsonl"

    # token files made by prepare_data.py
    data_dir = resolve(cfg["data_dir"])
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    torch.manual_seed(t["seed"])
    if device == "cuda":
        # faster matmuls on nvidia cards
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_cfg = GPTConfig(**cfg["model"])
    model = GPT(model_cfg).to(device)
    optimizer = model.configure_optimizer(t["weight_decay"], t["lr"], (t["beta1"], t["beta2"]))
    # mixed precision only on cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # its own generator for picking batches so resume lands on the same data
    batch_gen = torch.Generator().manual_seed(t["seed"])

    step, best_val, tokens_seen, elapsed_before = 0, float("inf"), 0, 0.0
    if resume and ckpt_path.exists():
        # load the saved run and put every piece of state back
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        best_val = ckpt["best_val"]
        tokens_seen = ckpt["tokens_seen"]
        elapsed_before = ckpt["elapsed"]
        torch.set_rng_state(ckpt["rng_torch"].cpu())
        if device == "cuda" and ckpt["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        batch_gen.set_state(ckpt["rng_batch"])
        if not quiet:
            print(f"resumed from step {step}")
    else:
        # fresh run, start the metrics file over
        metrics_path.write_text("")

    # optional and off by default, it is flaky on windows
    run_model = torch.compile(model) if t["compile"] else model

    if not quiet:
        print(f"device {device} | params {model.num_params() / 1e6:.2f}M | steps {t['max_steps']}")

    tokens_per_step = t["batch_size"] * t["grad_accum"] * model_cfg.block_size
    start = time.time()
    window_start, window_tokens = time.time(), 0
    last_val = None
    model.train()

    def checkpoint():
        elapsed = elapsed_before + time.time() - start
        save_checkpoint(ckpt_path, model, optimizer, scaler, step, batch_gen, cfg, best_val, tokens_seen, elapsed)

    def log(row):
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    while step < t["max_steps"]:
        # lets tests and impatient people stop early without losing work
        if stop_at is not None and step >= stop_at:
            break

        # check the validation loss every so often and save
        if step % t["eval_interval"] == 0 and step > 0:
            last_val = estimate_loss(model, val_data, t, model_cfg.block_size, device, use_amp)
            best_val = min(best_val, last_val)
            checkpoint()
            # don't count eval time against tokens per second
            window_start, window_tokens = time.time(), 0

        # set this step's learning rate
        lr = get_lr(step, t)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # gradient accumulation: several small batches act like one big one
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(t["grad_accum"]):
            x, y = get_batch(train_data, t["batch_size"], model_cfg.block_size, batch_gen, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss, _ = run_model(x, y)
            scaler.scale(loss / t["grad_accum"]).backward()
            loss_sum += loss.item()

        # clip so one bad batch can't blow up the weights
        if t["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_seen += tokens_per_step
        window_tokens += tokens_per_step

        # write a line to metrics.jsonl
        if step % t["log_interval"] == 0 or step == t["max_steps"]:
            tps = window_tokens / max(1e-9, time.time() - window_start)
            row = {
                "step": step,
                "train_loss": round(loss_sum / t["grad_accum"], 5),
                "val_loss": None if last_val is None else round(last_val, 5),
                "lr": lr,
                "tokens_per_sec": round(tps, 1),
                "tokens_seen": tokens_seen,
                "elapsed": round(elapsed_before + time.time() - start, 1),
            }
            log(row)
            if not quiet:
                val_txt = "" if last_val is None else f" | val {last_val:.4f}"
                print(f"step {step:>6} | train {row['train_loss']:.4f}{val_txt} | lr {lr:.2e} | {tps / 1000:.1f}k tok/s")
            last_val = None
            window_start, window_tokens = time.time(), 0

    # final validation number once the run is really done
    final_val = None
    if step >= t["max_steps"]:
        final_val = estimate_loss(model, val_data, t, model_cfg.block_size, device, use_amp)
        best_val = min(best_val, final_val)
        log({"step": step, "train_loss": None, "val_loss": round(final_val, 5), "lr": get_lr(step, t),
             "tokens_per_sec": None, "tokens_seen": tokens_seen,
             "elapsed": round(elapsed_before + time.time() - start, 1)})
        if not quiet:
            print(f"done | final val {final_val:.4f}")
    checkpoint()
    return model, final_val


# loads a trained model from a checkpoint file
def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]["model"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt
