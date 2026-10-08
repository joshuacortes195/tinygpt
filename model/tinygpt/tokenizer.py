import json
from collections import Counter, defaultdict

import regex as re

# gpt-2 style splitting rule, kept in the json so the browser uses the exact same one
GPT2_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?\p{L}+| ?\p{N}+"
    r"| ?[^\p{White_Space}\p{L}\p{N}]+"
    r"|\p{White_Space}+(?!\P{White_Space})"
    r"|\p{White_Space}+"
)

# marks the end of one story in the dataset
EOT = "<|endoftext|>"


class BPETokenizer:
    def __init__(self, merges=None, special_tokens=None, pattern=GPT2_PATTERN):
        self.merges = [tuple(m) for m in (merges or [])]
        self.special_tokens = dict(special_tokens or {})
        self.pattern = pattern
        self._build()

    # rebuilds the lookup tables from the merge list
    def _build(self):
        self.regex = re.compile(self.pattern)
        # ids 0-255 are the raw bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.ranks = {}
        # every merge makes one new token out of two older ones
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
            self.ranks[(a, b)] = 256 + i
        self.special_ids = {i: s for s, i in self.special_tokens.items()}
        # splits text around special tokens, longest first so none get cut short
        if self.special_tokens:
            names = sorted(self.special_tokens, key=len, reverse=True)
            self.special_regex = re.compile("(" + "|".join(re.escape(s) for s in names) + ")")
        else:
            self.special_regex = None
        # remembers chunks we already encoded
        self.cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special_tokens)

    # learns the merges from text
    @classmethod
    def train(cls, text, vocab_size, special_tokens=(EOT,), pattern=GPT2_PATTERN, verbose=False):
        num_merges = vocab_size - 256 - len(special_tokens)
        assert num_merges >= 0, "vocab size is too small"
        splitter = re.compile(pattern)

        # count each unique chunk once instead of walking the raw text every merge
        chunk_counts = Counter()
        special_split = "|".join(re.escape(s) for s in special_tokens)
        parts = re.split(special_split, text) if special_tokens else [text]
        for part in parts:
            chunk_counts.update(splitter.findall(part))

        # each unique chunk becomes a list of byte ids plus how often it shows up
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        counts = list(chunk_counts.values())

        # how often each pair shows up, and which words contain it
        pair_counts = defaultdict(int)
        pair_words = defaultdict(set)
        for wi, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += counts[wi]
                pair_words[pair].add(wi)

        merges = []
        for i in range(num_merges):
            if not pair_counts:
                break
            # pick the most common pair, ties go to the bigger pair so it is repeatable
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] <= 0:
                break
            new_id = 256 + i
            merges.append(best)

            # only touch the words that actually contain this pair
            for wi in list(pair_words[best]):
                word = words[wi]
                # take this word's old pairs out of the counts
                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= counts[wi]
                    pair_words[pair].discard(wi)
                # glue the pair together everywhere in the word
                merged = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j + 1]) == best:
                        merged.append(new_id)
                        j += 2
                    else:
                        merged.append(word[j])
                        j += 1
                words[wi] = merged
                # put the word's new pairs back into the counts
                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] += counts[wi]
                    pair_words[pair].add(wi)

            del pair_counts[best]
            del pair_words[best]
            if verbose and (i + 1) % 500 == 0:
                print(f"merge {i + 1}/{num_merges}")

        # special tokens take the ids right after the merges
        specials = {s: 256 + len(merges) + k for k, s in enumerate(special_tokens)}
        return cls(merges, specials, pattern)

    # turns one chunk into token ids by replaying the merges in the order they were learned
    def _encode_chunk(self, chunk):
        cached = self.cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # find the pair that was learned earliest
            best = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            new_id = self.ranks[best]
            merged = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and (ids[j], ids[j + 1]) == best:
                    merged.append(new_id)
                    j += 2
                else:
                    merged.append(ids[j])
                    j += 1
            ids = merged
        # keeps the cache from growing forever on weird text
        if len(self.cache) < 500_000:
            self.cache[chunk] = ids
        return ids

    # text to token ids
    def encode(self, text, allow_special=True):
        if allow_special and self.special_regex is not None:
            parts = self.special_regex.split(text)
        else:
            parts = [text]
        ids = []
        for part in parts:
            if not part:
                continue
            # special tokens map straight to their id
            if allow_special and part in self.special_tokens:
                ids.append(self.special_tokens[part])
                continue
            for chunk in self.regex.findall(part):
                ids.extend(self._encode_chunk(chunk))
        return ids

    # token ids back to text
    def decode(self, ids):
        out = []
        for i in ids:
            if i in self.special_ids:
                out.append(self.special_ids[i].encode("utf-8"))
            else:
                out.append(self.vocab[i])
        # half finished utf-8 characters become the replacement symbol instead of crashing
        return b"".join(out).decode("utf-8", errors="replace")

    # the raw bytes behind one token, used for showing tokens on the site
    def token_bytes(self, i):
        if i in self.special_ids:
            return self.special_ids[i].encode("utf-8")
        return self.vocab[i]

    def save(self, path):
        data = {
            "version": 1,
            "pattern": self.pattern,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["merges"], data["special_tokens"], data["pattern"])
