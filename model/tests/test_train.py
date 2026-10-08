import copy
import json

import numpy as np
import torch

from tinygpt.trainer import get_lr, load_model, train


# writes a small fake dataset and returns a config that points at it
def make_cfg(tmp_path, name):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    # a repeating pattern so there is something to learn
    pattern = rng.integers(0, 64, size=50)
    np.tile(pattern, 200).astype(np.uint16).tofile(data_dir / "train.bin")
    np.tile(pattern, 40).astype(np.uint16).tofile(data_dir / "val.bin")
    return {
        "name": name,
        "data_dir": str(data_dir),
        "out_dir": str(tmp_path / name),
        "model": {"vocab_size": 64, "block_size": 16, "n_layer": 2, "n_head": 2, "n_embd": 32, "dropout": 0.1},
        "train": {
            "batch_size": 8, "grad_accum": 2, "max_steps": 10, "lr": 1e-3, "min_lr": 1e-4,
            "warmup_steps": 2, "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
            "eval_interval": 5, "eval_batches": 2, "log_interval": 1, "seed": 7,
            "compile": False, "device": "cpu",
        },
    }


def test_lr_schedule():
    t = {"lr": 1.0, "min_lr": 0.1, "warmup_steps": 10, "max_steps": 110}
    assert get_lr(0, t) < get_lr(5, t) < get_lr(9, t) <= 1.0
    assert abs(get_lr(10, t) - 1.0) < 1e-9
    assert get_lr(60, t) < get_lr(20, t)
    assert abs(get_lr(110, t) - 0.1) < 1e-9


def test_loss_goes_down_and_metrics_are_logged(tmp_path):
    cfg = make_cfg(tmp_path, "run")
    cfg["train"]["max_steps"] = 60
    cfg["train"]["eval_interval"] = 20
    train(cfg, quiet=True)
    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    train_rows = [r for r in rows if r["train_loss"] is not None]
    assert train_rows[-1]["train_loss"] < train_rows[0]["train_loss"]
    assert any(r["val_loss"] is not None for r in rows)
    assert {"step", "train_loss", "val_loss", "lr", "tokens_per_sec"} <= set(rows[0])


# stopping half way and resuming must give the exact same weights as never stopping
def test_resume_matches_uninterrupted_run(tmp_path):
    straight, _ = train(make_cfg(tmp_path, "straight"), quiet=True)

    cfg = make_cfg(tmp_path, "resumed")
    train(copy.deepcopy(cfg), stop_at=5, quiet=True)
    resumed, _ = train(copy.deepcopy(cfg), resume=True, quiet=True)

    for a, b in zip(straight.state_dict().values(), resumed.state_dict().values()):
        assert torch.equal(a, b)


def test_checkpoint_loads(tmp_path):
    cfg = make_cfg(tmp_path, "run")
    trained, _ = train(cfg, quiet=True)
    loaded, ckpt = load_model(tmp_path / "run" / "ckpt.pt")
    assert ckpt["step"] == 10
    x = torch.randint(0, 64, (1, 8))
    trained.eval()
    assert torch.equal(trained(x)[0], loaded(x)[0])
