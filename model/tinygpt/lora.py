import math

import torch
import torch.nn as nn


# wraps a frozen linear layer and learns a small low-rank update next to it
class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        # the original weights never change
        for p in self.base.parameters():
            p.requires_grad = False
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        # the update is B @ A, two thin matrices instead of one full sized one
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        # A starts random and B starts at zero, so at step 0 the model is exactly the original
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        update = self.dropout(x) @ self.A.T @ self.B.T
        return self.base(x) + update * self.scale

    # folds the update into the original weight so the result is a plain linear layer again
    @torch.no_grad()
    def merged(self):
        self.base.weight += (self.B @ self.A).to(self.base.weight.dtype) * self.scale
        return self.base


# swaps the named linear layers inside a model for lora versions
def add_lora(model, targets=("q_proj", "v_proj"), rank=8, alpha=16, dropout=0.05):
    # freeze everything first, only the lora matrices will train
    for p in model.parameters():
        p.requires_grad = False
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
                count += 1
    return count


# puts every lora update back into its layer and removes the wrappers
def merge_lora(model):
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
    return model


# just the trained matrices, a few MB instead of the whole model
def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}
