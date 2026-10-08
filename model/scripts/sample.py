"""Writes text with a trained checkpoint."""
import argparse
from pathlib import Path

import torch

from tinygpt.tokenizer import BPETokenizer, EOT
from tinygpt.trainer import load_model, pick_device, resolve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="path to a ckpt.pt file")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--tokenizer", default=None, help="defaults to the tokenizer in the run's data_dir")
    args = p.parse_args()

    device = pick_device()
    model, ckpt = load_model(args.checkpoint, device)
    tok_path = Path(args.tokenizer) if args.tokenizer else resolve(ckpt["config"]["data_dir"]) / "tokenizer.json"
    tok = BPETokenizer.load(tok_path)

    # seeded so the same command gives the same story
    g = torch.Generator(device=device).manual_seed(args.seed)
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    for _ in range(args.num_samples):
        out = model.generate(
            ids, args.max_tokens, args.temperature, args.top_k, args.top_p,
            generator=g, stop_token=tok.special_tokens.get(EOT),
        )
        print(tok.decode(out[0].tolist()))
        print("-" * 40)


if __name__ == "__main__":
    main()
