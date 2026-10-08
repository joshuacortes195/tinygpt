"""Exports the LoRA fine-tuned model to a folder Transformers.js can load in the browser."""
import argparse
import json
import shutil
from pathlib import Path

import onnx
from onnx.external_data_helper import set_external_data
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.exporters.onnx import main_export

MODEL_ROOT = Path(__file__).resolve().parent.parent
WEB_MODELS = MODEL_ROOT.parent / "web" / "public" / "models"

# small files the browser needs next to the model
KEEP = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]


# moves the weights out of the onnx file into several data files that each fit under github's limit
def chunk_weights(src, dest, chunk_bytes=90_000_000):
    model = onnx.load(str(src))
    chunk, used = 0, 0
    for tensor in model.graph.initializer:
        size = len(tensor.raw_data)
        # tiny tensors can stay inside the main file
        if size < 1024:
            continue
        if used and used + size > chunk_bytes:
            chunk, used = chunk + 1, 0
        # transformers.js looks for <file>_data, <file>_data_1, <file>_data_2 ...
        location = f"{dest.name}_data" + (f"_{chunk}" if chunk else "")
        set_external_data(tensor, location=location)
        used += size
    onnx.save_model(model, str(dest))
    return chunk + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/smollm2-lora", help="folder written by finetune_lora.py")
    p.add_argument("--id", default="smollm2-stories")
    p.add_argument("--name", default="SmolLM2-135M + LoRA")
    args = p.parse_args()

    run = MODEL_ROOT / args.run
    work = run / "onnx"
    out = WEB_MODELS / args.id
    if out.exists():
        shutil.rmtree(out)
    (out / "onnx").mkdir(parents=True)

    # step 1: pytorch to onnx, with the key/value cache inputs so generation is fast
    main_export(str(run / "merged"), output=str(work), task="text-generation-with-past")

    # step 2: 8 bit weights, about a quarter of the size
    quantized = work / "model_quantized.onnx"
    quantize_dynamic(str(work / "model.onnx"), str(quantized), weight_type=QuantType.QUInt8)

    # step 3: split the weights into files small enough to host
    chunks = chunk_weights(quantized, out / "onnx" / "model_quantized.onnx")

    for name in KEEP:
        if (work / name).exists():
            shutil.copyfile(work / name, out / name)

    # tell transformers.js how many data files to fetch
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    config["transformers.js_config"] = {"use_external_data_format": {"model_quantized.onnx": chunks}}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in (out / "onnx").iterdir())
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["download_bytes"] = total
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # list it in the manifest as an optional extra, separate from the from-scratch models
    manifest_path = WEB_MODELS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = {
        "id": args.id,
        "name": args.name,
        "description": "A pretrained 135M model, fine-tuned on the same stories with LoRA. Not built from scratch, here for comparison.",
        "path": args.id,
        "dtype": "q8",
        "bytes": total,
    }
    manifest["extras"] = [e for e in manifest.get("extras", []) if e["id"] != args.id] + [entry]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out} | {chunks} data files | {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
