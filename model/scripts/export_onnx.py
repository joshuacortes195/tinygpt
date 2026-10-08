"""Exports a checkpoint to a model folder the website can load."""
import argparse
import json
from pathlib import Path

import numpy as np

from tinygpt.export import add_to_manifest, export_model_folder
from tinygpt.trainer import load_model, resolve

MODEL_ROOT = Path(__file__).resolve().parent.parent
WEB_MODELS = MODEL_ROOT.parent / "web" / "public" / "models"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="path to a ckpt.pt file")
    p.add_argument("--id", required=True, help="folder name under web/public/models, e.g. v0-smoke")
    p.add_argument("--name", default=None, help="name shown in the model picker")
    p.add_argument("--description", default="")
    p.add_argument("--out", default=None, help="where to write the folder, defaults to web/public/models/<id>")
    p.add_argument("--no-manifest", action="store_true", help="don't touch manifest.json")
    p.add_argument("--default", action="store_true", help="make this the model the site opens with")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    model, ckpt = load_model(ckpt_path)
    data_dir = resolve(ckpt["config"]["data_dir"])
    out_dir = Path(args.out) if args.out else WEB_MODELS / args.id
    name = args.name or args.id

    # real validation tokens make the int8 check meaningful
    samples = None
    if (data_dir / "val.bin").exists():
        samples = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")[:500_000]

    config = export_model_folder(
        model, ckpt, data_dir / "tokenizer.json", ckpt_path.parent / "metrics.jsonl",
        out_dir, name, args.description, samples,
    )
    print(json.dumps(config, indent=2))

    # fp32 has to match pytorch almost exactly or the export is broken
    assert config["parity"]["fp32_max_logit_diff"] < 1e-3, "fp32 export does not match pytorch"

    if not args.no_manifest:
        add_to_manifest(WEB_MODELS / "manifest.json", args.id, name, args.description, args.default)
        print(f"added {args.id} to manifest.json")


if __name__ == "__main__":
    main()
