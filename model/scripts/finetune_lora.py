"""LoRA fine-tunes a small pretrained model on TinyStories, as a comparison for the from-scratch models.

This is the one place in the project that uses Hugging Face transformers. It only loads the
pretrained model and its tokenizer. The LoRA layers and the training loop are my own.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinygpt.lora import add_lora, lora_state_dict, merge_lora
from tinygpt.tokenizer import EOT
from tinygpt.trainer import pick_device

MODEL_ROOT = Path(__file__).resolve().parent.parent


# reads the first max_mb of a text file and turns it into one long array of token ids
def tokenize_stories(path, tok, max_mb):
    with open(path, "rb") as f:
        text = f.read(int(max_mb * 1e6)).decode("utf-8", errors="ignore")
    stories = [s.strip() for s in text.split(EOT) if s.strip()]
    ids = []
    # tokenize in batches, each story ends with the model's own end token
    for i in range(0, len(stories), 2000):
        for story_ids in tok(stories[i : i + 2000], add_special_tokens=False)["input_ids"]:
            ids.extend(story_ids)
            ids.append(tok.eos_token_id)
    return np.array(ids, dtype=np.int32), len(text.encode("utf-8"))


# random windows of tokens, hugging face shifts them by one itself to make the targets
def get_batch(data, batch_size, block_size, generator, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator).tolist()
    return torch.from_numpy(np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)).to(device)


# average loss on a fixed set of validation batches
@torch.no_grad()
def evaluate(model, data, args, device, use_amp):
    model.eval()
    g = torch.Generator().manual_seed(args.seed + 1)
    losses = []
    for _ in range(args.eval_batches):
        x = get_batch(data, args.batch_size, args.block_size, g, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = model(input_ids=x, labels=x).loss
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--out", default="runs/smollm2-lora")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--train-mb", type=float, default=60, help="how much story text to fine-tune on")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=30)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    use_amp = device == "cuda"
    out_dir = MODEL_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = MODEL_ROOT / args.data_dir / "raw"
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print("tokenizing stories")
    train_data, _ = tokenize_stories(raw / "TinyStoriesV2-GPT4-train.txt", tok, args.train_mb)
    val_data, val_bytes = tokenize_stories(raw / "TinyStoriesV2-GPT4-valid.txt", tok, 5)
    print(f"train tokens {len(train_data):,} | val tokens {len(val_data):,}")

    # how the untouched pretrained model does on stories, the "before" number
    base_val = evaluate(model, val_data, args, device, use_amp)
    print(f"pretrained val loss {base_val:.4f}")

    # add the small trainable matrices to the attention layers
    wrapped = add_lora(model, rank=args.rank, alpha=args.alpha)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"lora on {wrapped} layers | training {n_trainable:,} of {total_params:,} params ({100 * n_trainable / total_params:.2f}%)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    g = torch.Generator().manual_seed(args.seed)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")

    def log(row):
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    log({"step": 0, "train_loss": None, "val_loss": round(base_val, 5)})
    start = time.time()
    model.train()
    last_val = base_val
    for step in range(1, args.max_steps + 1):
        # warmup then cosine decay, same shape as the from-scratch runs
        if step <= args.warmup_steps:
            lr = args.lr * step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr

        x = get_batch(train_data, args.batch_size, args.block_size, g, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = model(input_ids=x, labels=x).loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0:
            log({"step": step, "train_loss": round(loss.item(), 5), "val_loss": None, "lr": lr})
            print(f"step {step:>5} | train {loss.item():.4f} | lr {lr:.2e} | {time.time() - start:.0f}s")
        if step % args.eval_interval == 0 or step == args.max_steps:
            last_val = evaluate(model, val_data, args, device, use_amp)
            log({"step": step, "train_loss": None, "val_loss": round(last_val, 5)})
            print(f"step {step:>5} | val {last_val:.4f}")

    elapsed = time.time() - start

    # keep the tiny adapter on its own, then fold it into the model for export
    torch.save(lora_state_dict(model), out_dir / "lora.pt")
    merge_lora(model)
    model.eval()
    model.save_pretrained(out_dir / "merged")
    tok.save_pretrained(out_dir / "merged")

    summary = {
        "base_model": args.model,
        "total_params": total_params,
        "trainable_params": n_trainable,
        "rank": args.rank,
        "steps": args.max_steps,
        "training_tokens": args.max_steps * args.batch_size * args.block_size,
        "training_seconds": round(elapsed, 1),
        "pretrained_val_loss": round(base_val, 5),
        "finetuned_val_loss": round(last_val, 5),
        # tokens per byte lets this be compared with models that use a different tokenizer
        "val_tokens_per_byte": len(val_data) / val_bytes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
