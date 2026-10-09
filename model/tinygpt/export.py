import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# what the browser calls: token ids in, scores for the next token and the attention maps out
class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        x, attn = self.model.hidden(input_ids, return_attn=True)
        # only the last position matters for picking the next token
        last = x[:, -1, :]
        # multiply by the embedding table directly so the file stores the tied weights once
        logits = (self.model.wte.weight @ last.transpose(0, 1)).transpose(0, 1)
        return logits, attn


# writes the fp32 onnx file, sequence length stays flexible
def export_fp32(model, path, opset=17):
    model = model.cpu().eval()
    example = torch.randint(0, model.config.vocab_size, (1, min(8, model.config.block_size)))
    torch.onnx.export(
        ExportWrapper(model),
        (example,),
        str(path),
        input_names=["input_ids"],
        output_names=["logits", "attention"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
            "attention": {0: "batch", 3: "seq", 4: "seq"},
        },
        opset_version=opset,
        dynamo=False,
    )


# makes a smaller int8 copy, weights are stored as 8 bit numbers
def export_int8(fp32_path, int8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)


def _session(path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


# compares pytorch against the onnx files on the same inputs
def check_parity(model, fp32_path, int8_path=None, samples=None, n=20, seed=0):
    model = model.cpu().eval()
    rng = np.random.default_rng(seed)
    cfg = model.config
    fp32 = _session(fp32_path)
    int8 = _session(int8_path) if int8_path else None

    max_logit_diff, max_attn_diff = 0.0, 0.0
    int8_agree, int8_max_diff = 0, 0.0
    for i in range(n):
        # try lots of different lengths, including the shortest and the longest
        T = [1, cfg.block_size][i] if i < 2 else int(rng.integers(2, cfg.block_size + 1))
        if samples is not None:
            # real text is a fairer test for int8 than random ids
            start = int(rng.integers(0, len(samples) - T))
            ids = np.asarray(samples[start : start + T], dtype=np.int64)[None, :]
        else:
            ids = rng.integers(0, cfg.vocab_size, size=(1, T), dtype=np.int64)

        with torch.no_grad():
            logits, _, attn = model(torch.from_numpy(ids), return_attn=True)
        want = logits[:, -1, :].numpy()

        got, got_attn = fp32.run(None, {"input_ids": ids})
        max_logit_diff = max(max_logit_diff, float(np.abs(got - want).max()))
        max_attn_diff = max(max_attn_diff, float(np.abs(got_attn - attn.numpy()).max()))

        if int8 is not None:
            q = int8.run(["logits"], {"input_ids": ids})[0]
            int8_agree += int(q.argmax() == want.argmax())
            int8_max_diff = max(int8_max_diff, float(np.abs(q - want).max()))

    report = {"fp32_max_logit_diff": max_logit_diff, "fp32_max_attention_diff": max_attn_diff, "inputs": n}
    if int8 is not None:
        report["int8_top1_agreement"] = int8_agree / n
        report["int8_max_logit_diff"] = int8_max_diff
    return report


# github refuses files over 100 MB, so big files get cut into parts the site glues back together
def split_file(path, part_bytes=45_000_000):
    path = Path(path)
    size = path.stat().st_size
    if size <= part_bytes:
        return {"file": path.name, "bytes": size}
    parts = []
    with open(path, "rb") as f:
        while True:
            chunk = f.read(part_bytes)
            if not chunk:
                break
            name = f"{path.name}.part{len(parts)}"
            (path.parent / name).write_bytes(chunk)
            parts.append(name)
    # the whole file is not needed once the parts exist
    path.unlink()
    return {"parts": parts, "bytes": size}


# shrinks metrics.jsonl down to something a chart can load fast
def downsample_metrics(metrics_path, max_points=200):
    rows = []
    if Path(metrics_path).exists():
        for line in Path(metrics_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    train = [r for r in rows if r.get("train_loss") is not None]
    stride = max(1, len(train) // max_points)
    kept = train[::stride]
    # always keep the very last point
    if train and kept[-1] is not train[-1]:
        kept.append(train[-1])
    return {
        "train": [{"step": r["step"], "loss": r["train_loss"], "lr": r["lr"]} for r in kept],
        "val": [{"step": r["step"], "loss": r["val_loss"]} for r in rows if r.get("val_loss") is not None],
        "tokens_per_sec": float(np.median([r["tokens_per_sec"] for r in train])) if train else None,
    }


# writes the whole model folder the website loads
def export_model_folder(model, ckpt, tokenizer_path, metrics_path, out_dir, name, description="", samples=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path, int8_path = out_dir / "model.onnx", out_dir / "model.int8.onnx"

    export_fp32(model, fp32_path)
    export_int8(fp32_path, int8_path)
    parity = check_parity(model, fp32_path, int8_path, samples=samples)

    shutil.copyfile(tokenizer_path, out_dir / "tokenizer.json")
    metrics = downsample_metrics(metrics_path)
    (out_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    cfg = model.config
    val_points = metrics["val"]
    config = {
        "name": name,
        "description": description,
        "vocab_size": cfg.vocab_size,
        "block_size": cfg.block_size,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "params": model.num_params(),
        "training_tokens": ckpt.get("tokens_seen"),
        "training_steps": ckpt.get("step"),
        "training_seconds": ckpt.get("elapsed"),
        "final_val_loss": val_points[-1]["loss"] if val_points else None,
        "has_attention": True,
        "files": {"fp32": split_file(fp32_path), "int8": split_file(int8_path)},
        "parity": parity,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


# adds the model to manifest.json, or updates it if it is already listed
def add_to_manifest(manifest_path, model_id, name, description, make_default=False, examples=None):
    manifest_path = Path(manifest_path)
    manifest = {"default": model_id, "models": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = {"id": model_id, "name": name, "description": description, "path": model_id}
    # starting prompts the site offers for this model
    if examples:
        entry["examples"] = list(examples)
    manifest["models"] = [m for m in manifest["models"] if m["id"] != model_id] + [entry]
    if make_default:
        manifest["default"] = model_id
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
