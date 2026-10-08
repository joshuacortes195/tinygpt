"""Downloads TinyStories, trains the tokenizer and writes train.bin / val.bin."""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tinygpt.tokenizer import BPETokenizer, EOT

MODEL_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}


# downloads one file in pieces so it never sits in ram
def download(url, dest):
    if dest.exists():
        print(f"already have {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))
        bar.close()
    tmp.rename(dest)


# reads the first max_bytes of a text file
def read_sample(path, max_bytes):
    with open(path, "rb") as f:
        raw = f.read(max_bytes)
    # a cut in the middle of a character just gets dropped
    return raw.decode("utf-8", errors="ignore")


# yields the file in pieces that always end on a story boundary
def story_chunks(path, chunk_bytes, max_bytes=None):
    leftover = ""
    read = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            if max_bytes is not None and read >= max_bytes:
                break
            piece = f.read(chunk_bytes)
            if not piece:
                break
            read += len(piece)
            text = leftover + piece
            # hold back the unfinished story for the next piece
            cut = text.rfind(EOT)
            if cut == -1:
                leftover = text
                continue
            cut += len(EOT)
            leftover = text[cut:]
            yield text[:cut]
    if leftover.strip():
        yield leftover


# tokenizes a text file straight into a uint16 file on disk
def encode_file(tok, src, dest, chunk_bytes, max_bytes=None):
    total = 0
    size = src.stat().st_size if max_bytes is None else min(max_bytes, src.stat().st_size)
    bar = tqdm(total=size, unit="B", unit_scale=True, desc=f"encode {dest.name}")
    with open(dest, "wb") as out:
        for text in story_chunks(src, chunk_bytes, max_bytes):
            ids = np.array(tok.encode(text), dtype=np.uint16)
            ids.tofile(out)
            total += len(ids)
            bar.update(len(text))
    bar.close()
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--tokenizer-sample-mb", type=float, default=50, help="how much text the tokenizer learns from")
    p.add_argument("--max-train-mb", type=float, default=None, help="only tokenize this much of the train set")
    p.add_argument("--chunk-mb", type=float, default=8)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = MODEL_ROOT / data_dir
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # step 1: get the dataset
    if not args.skip_download:
        for name in FILES.values():
            download(BASE_URL + name, raw_dir / name)
    train_txt, val_txt = raw_dir / FILES["train"], raw_dir / FILES["val"]

    # step 2: train the tokenizer on a sample, or reuse the one we already have
    tok_path = data_dir / "tokenizer.json"
    if tok_path.exists():
        print("already have tokenizer.json")
        tok = BPETokenizer.load(tok_path)
    else:
        sample = read_sample(train_txt, int(args.tokenizer_sample_mb * 1e6))
        print(f"training tokenizer on {len(sample) / 1e6:.0f} MB of text")
        tok = BPETokenizer.train(sample, args.vocab_size, verbose=True)
        tok.save(tok_path)
        del sample
    # uint16 can only hold ids up to 65535
    assert tok.vocab_size <= 65536

    # step 3: tokenize everything to disk
    chunk = int(args.chunk_mb * 1e6)
    max_bytes = None if args.max_train_mb is None else int(args.max_train_mb * 1e6)
    n_train = encode_file(tok, train_txt, data_dir / "train.bin", chunk, max_bytes)
    n_val = encode_file(tok, val_txt, data_dir / "val.bin", chunk)
    print(f"train tokens: {n_train:,} | val tokens: {n_val:,} | vocab: {tok.vocab_size}")


if __name__ == "__main__":
    main()
