import torch
import torch.nn as nn

from tinygpt.lora import LoRALinear, add_lora, lora_state_dict, merge_lora


# a stand-in for one attention layer of a bigger model
class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.other = nn.Linear(16, 4)

    def forward(self, x):
        return self.other(self.q_proj(x) + self.v_proj(x))


def test_starts_identical_to_the_original():
    torch.manual_seed(0)
    model = Toy()
    x = torch.randn(3, 16)
    before = model(x)
    assert add_lora(model, rank=4) == 2
    model.eval()
    assert torch.allclose(model(x), before, atol=1e-6)


def test_only_lora_params_train():
    model = Toy()
    add_lora(model, rank=4)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith(".A") or n.endswith(".B") for n in trainable)
    # 2 layers, each with A (4x16) and B (16x4)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 2 * (4 * 16 + 16 * 4)


def test_merge_gives_the_same_outputs_with_plain_layers():
    torch.manual_seed(0)
    model = Toy()
    add_lora(model, rank=4, dropout=0.0)
    # pretend some training happened
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.B, std=0.1)
    x = torch.randn(3, 16)
    model.eval()
    want = model(x)
    merge_lora(model)
    assert isinstance(model.q_proj, nn.Linear) and isinstance(model.v_proj, nn.Linear)
    assert torch.allclose(model(x), want, atol=1e-5)


def test_state_dict_has_only_the_small_matrices():
    model = Toy()
    add_lora(model, rank=4)
    keys = list(lora_state_dict(model))
    assert len(keys) == 4 and all(k.endswith((".A", ".B")) for k in keys)
