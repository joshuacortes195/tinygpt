import torch
import torch.nn.functional as F


# picks the next token from the scores of the last position
def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    # temperature 0 means always take the most likely token
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    # lower temperature makes the model more sure of itself
    logits = logits / temperature

    # top-k: keep only the k best tokens
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = F.softmax(logits, dim=-1)

    # top-p: keep the smallest set of tokens whose chances add up to p
    if top_p is not None and 0 < top_p < 1:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # drop a token if the ones before it already cover p
        drop = (cumulative - sorted_probs) >= top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1, generator=generator)
