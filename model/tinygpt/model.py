import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# the knobs that decide the size of the model
@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # one layer makes the queries, keys and values together
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        # mixes the heads back into one vector
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # lower triangle of ones, so a token can only look at itself and earlier tokens
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # reshape to (batch, heads, time, head_dim) so every head works on its own
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # how much each token matches every other token
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # hide the future so the model can't cheat
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        # turn the scores into weights that add up to 1
        att = F.softmax(att, dim=-1)
        weights = att
        att = self.attn_dropout(att)

        # weighted mix of the values, then stitch the heads back together
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y, weights


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # widen, bend with gelu, squeeze back down
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # pre-layernorm: normalize first, then add the result back on
        a, weights = self.attn(self.ln1(x))
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, weights


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # token ids to vectors
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # learned vector for each position
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: the output layer shares the embedding table
        self.head.weight = self.wte.weight

        self.apply(self._init_weights)
        # smaller start for the layers that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # tied weights are only counted once here
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # everything except the output layer: token ids in, one vector per position out
    def hidden(self, idx, return_attn=False):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence is longer than the context window"
        pos = torch.arange(T, device=idx.device)
        # token meaning plus position
        x = self.drop(self.wte(idx) + self.wpe(pos))

        attns = []
        for block in self.blocks:
            x, weights = block(x)
            if return_attn:
                attns.append(weights)

        x = self.ln_f(x)
        # attention comes back as (batch, layers, heads, time, time)
        attn = torch.stack(attns, dim=1) if return_attn else None
        return x, attn

    def forward(self, idx, targets=None, return_attn=False):
        x, attn = self.hidden(idx, return_attn)
        # a score for every token in the vocab at every position
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, attn

    # weight decay only on the big matrices, not on biases or layernorm
    def configure_optimizer(self, weight_decay, lr, betas):
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    # writes new tokens one at a time
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, generator=None, stop_token=None):
        from .sampling import sample_next

        for _ in range(max_new_tokens):
            # only the last block_size tokens fit in the context
            idx_cond = idx[:, -self.config.block_size :]
            logits, _, _ = self(idx_cond)
            next_id = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_token is not None and (next_id == stop_token).all():
                break
        return idx
