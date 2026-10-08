import torch

from tinygpt.model import GPT, GPTConfig
from tinygpt.sampling import sample_next


def tiny_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32))


def test_output_shapes():
    model = tiny_model()
    x = torch.randint(0, 64, (3, 10))
    logits, loss, attn = model(x, targets=x, return_attn=True)
    assert logits.shape == (3, 10, 64)
    assert loss.ndim == 0
    # batch, layers, heads, time, time
    assert attn.shape == (3, 2, 2, 10, 10)


def test_no_attention_unless_asked():
    model = tiny_model()
    _, loss, attn = model(torch.randint(0, 64, (1, 5)))
    assert loss is None and attn is None


# changing a later token must not change anything before it
def test_causal_mask_blocks_the_future():
    model = tiny_model().eval()
    x = torch.randint(0, 64, (1, 12))
    y = x.clone()
    y[0, 8] = (y[0, 8] + 1) % 64
    a, _, _ = model(x)
    b, _, _ = model(y)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-6)
    assert not torch.allclose(a[0, 8:], b[0, 8:], atol=1e-6)


# each row of attention is a set of weights that sums to 1 and never looks ahead
def test_attention_weights():
    model = tiny_model().eval()
    _, _, attn = model(torch.randint(0, 64, (2, 9)), return_attn=True)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)
    future = torch.triu(torch.ones(9, 9, dtype=torch.bool), diagonal=1)
    assert attn[..., future].abs().max() == 0


def test_weight_tying():
    model = tiny_model()
    assert model.head.weight is model.wte.weight


# biases and layernorm must not get weight decay
def test_weight_decay_groups():
    model = tiny_model()
    opt = model.configure_optimizer(0.1, 1e-3, (0.9, 0.95))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    counted = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert counted == model.num_params()


# a healthy model can memorize one batch
def test_overfits_one_batch():
    model = tiny_model()
    opt = model.configure_optimizer(0.0, 3e-3, (0.9, 0.95))
    x = torch.randint(0, 64, (4, 16))
    y = torch.roll(x, -1, dims=1)
    first = None
    for _ in range(200):
        _, loss, _ = model(x, y)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * first


def test_generate_length_and_seed():
    model = tiny_model().eval()
    x = torch.randint(0, 64, (1, 4))
    a = model.generate(x, 30, temperature=0.9, top_k=20, generator=torch.Generator().manual_seed(1))
    b = model.generate(x, 30, temperature=0.9, top_k=20, generator=torch.Generator().manual_seed(1))
    # goes past block_size on purpose to check the context gets cropped
    assert a.shape == (1, 34)
    assert torch.equal(a, b)


def test_sampling_filters():
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0, -1.0]])
    g = torch.Generator().manual_seed(0)
    # temperature 0 is greedy
    assert sample_next(logits, temperature=0).item() == 0
    # top-k of 2 only ever picks the two best
    picks = {sample_next(logits, 1.0, top_k=2, generator=g).item() for _ in range(200)}
    assert picks <= {0, 1}
    # a tiny top-p keeps just the best token
    picks = {sample_next(logits, 1.0, top_p=0.1, generator=g).item() for _ in range(50)}
    assert picks == {0}
