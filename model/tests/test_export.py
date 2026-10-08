import json

import numpy as np
import pytest
import torch

from tinygpt.export import add_to_manifest, check_parity, downsample_metrics, export_fp32, export_int8
from tinygpt.model import GPT, GPTConfig

ort = pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32)).eval()
    out = tmp_path_factory.mktemp("export")
    export_fp32(model, out / "model.onnx")
    export_int8(out / "model.onnx", out / "model.int8.onnx")
    return model, out


# onnx has to give the same numbers as pytorch at every sequence length
def test_fp32_matches_pytorch(exported):
    model, out = exported
    report = check_parity(model, out / "model.onnx", out / "model.int8.onnx")
    assert report["fp32_max_logit_diff"] < 1e-4
    assert report["fp32_max_attention_diff"] < 1e-4
    assert 0.0 <= report["int8_top1_agreement"] <= 1.0


def test_dynamic_length_and_output_shapes(exported):
    _, out = exported
    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    for T in (1, 5, 16):
        ids = np.zeros((1, T), dtype=np.int64)
        logits, attn = sess.run(None, {"input_ids": ids})
        assert logits.shape == (1, 64)
        assert attn.shape == (1, 2, 2, T, T)


def test_int8_is_smaller(exported):
    _, out = exported
    assert (out / "model.int8.onnx").stat().st_size < (out / "model.onnx").stat().st_size


def test_downsample_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"step": i, "train_loss": 5.0 - i * 0.001, "val_loss": 4.0 if i % 100 == 0 else None, "lr": 1e-3, "tokens_per_sec": 1000.0}
        for i in range(1, 1001)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    out = downsample_metrics(path, max_points=100)
    assert len(out["train"]) <= 102
    assert out["train"][-1]["step"] == 1000
    assert len(out["val"]) == 10


def test_manifest_add_and_update(tmp_path):
    path = tmp_path / "manifest.json"
    add_to_manifest(path, "a", "A", "first")
    add_to_manifest(path, "b", "B", "second", make_default=True)
    m = add_to_manifest(path, "a", "A2", "renamed")
    assert m["default"] == "b"
    assert [x["id"] for x in m["models"]] == ["b", "a"]
    assert m["models"][1]["name"] == "A2"
