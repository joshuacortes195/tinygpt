"""Collects the most common words in recipe titles, the site uses them to fix typos in a dish name."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tinygpt.tokenizer import EOT

MODEL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = MODEL_ROOT.parent / "web" / "src" / "lib" / "dishWords.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=str(MODEL_ROOT / "data-recipes" / "raw" / "recipes-train.txt"))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--top", type=int, default=5000, help="how many words to keep")
    args = p.parse_args()

    counts = Counter()
    # the title is the first line of the file and every line right after an end marker
    is_title = True
    with open(args.text, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if is_title and line:
                # only plain words of 3 letters or more are worth correcting to
                counts.update(re.findall(r"[a-z]{3,}", line.lower()))
            is_title = line == EOT

    # most common first, so the site can prefer the likelier word when two are equally close
    words = [w for w, _ in counts.most_common(args.top)]
    Path(args.out).write_text(json.dumps(words), encoding="utf-8")
    print(f"wrote {len(words):,} words, the rarest one shows up {counts[words[-1]]:,} times")


if __name__ == "__main__":
    main()
