import json
from pathlib import Path

import pytest

from tinygpt.tokenizer import BPETokenizer, EOT

FIXTURES = Path(__file__).resolve().parents[2] / "shared" / "fixtures"

TEXT = (
    "Once upon a time there was a little girl named Lily. She loved to play outside. "
    "One day, Lily found a big red ball. \"Look, mom!\" she said. The ball was very bouncy. "
    "Lily and her mom played with the ball all day. They were happy.\n\n"
) * 20

TRICKY = [
    "",
    " ",
    "hello world",
    "  lots   of    spaces  ",
    "tabs\tand\nnewlines\r\n\n\n",
    "café naïve résumé",
    "emoji 😀🎉 and 👨‍👩‍👧 family",
    "日本語のテキスト",
    "don't can't I'll we've",
    "supercalifragilisticexpialidocious" * 3,
    "numbers 12345 and 3.14159",
    f"story one{EOT}story two",
]


# small tokenizer trained on the text above, shared by the tests
@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(TEXT, vocab_size=400)


def test_round_trip(tok):
    for text in TRICKY:
        assert tok.decode(tok.encode(text)) == text


def test_vocab_size(tok):
    assert tok.vocab_size <= 400
    assert all(0 <= i < tok.vocab_size for i in tok.encode(TEXT))


# merging should make the text shorter than its raw bytes
def test_compresses(tok):
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8")) / 2


def test_special_token(tok):
    ids = tok.encode(f"a{EOT}b")
    assert tok.special_tokens[EOT] in ids
    # with specials turned off it is just normal text
    plain = tok.encode(EOT, allow_special=False)
    assert tok.special_tokens[EOT] not in plain
    assert tok.decode(plain) == EOT


# bytes the tokenizer never saw in training still work
def test_no_unknown_tokens(tok):
    text = "ξ ∑ ☃ \x00 \x7f"
    assert tok.decode(tok.encode(text)) == text


def test_training_is_deterministic():
    a = BPETokenizer.train(TEXT, vocab_size=350)
    b = BPETokenizer.train(TEXT, vocab_size=350)
    assert a.merges == b.merges


def test_save_load(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    for text in TRICKY:
        assert loaded.encode(text) == tok.encode(text)


# the same golden cases are checked by the typescript tokenizer
def test_golden_vectors():
    cases_path = FIXTURES / "tokenizer_cases.json"
    if not cases_path.exists():
        pytest.skip("fixtures not generated yet")
    tok = BPETokenizer.load(FIXTURES / "tokenizer.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    assert len(cases) > 0
    for case in cases:
        assert tok.encode(case["text"]) == case["ids"], case["name"]
        assert tok.decode(case["ids"]) == case["text"], case["name"]
