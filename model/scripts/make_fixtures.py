"""Writes the golden tokenizer cases that both the python and typescript tests check."""
import json
import shutil
from pathlib import Path

from tinygpt.tokenizer import BPETokenizer, EOT

MODEL_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = MODEL_ROOT.parent / "shared" / "fixtures"

CASES = {
    "empty": "",
    "single space": " ",
    "simple": "Once upon a time, there was a little girl named Lily.",
    "leading and trailing spaces": "  hello world  ",
    "whitespace runs": "a    b\t\tc\n\n\nd \n e",
    "windows newlines": "line one\r\nline two\r\n",
    "accents": "café naïve résumé jalapeño über",
    "emoji": "I am happy 😀🎉 today!",
    "emoji with joiners": "family 👨‍👩‍👧‍👦 and flag 🇺🇸",
    "cjk": "日本語のテキスト と 中文",
    "contractions": "don't can't I'll we've she's they're I'm he'd",
    "long word": "supercalifragilisticexpialidocious" * 3,
    "numbers": "In 2024 there were 1234567 stars and 3.14159 pies.",
    "punctuation": "Wait... what?! \"Really,\" she said -- (yes) [no] {maybe}.",
    "dialogue": "\"Can I play?\" asked Tom. \"Yes!\" said his mom.\n\nThey played all day.",
    "special token": f"The end.{EOT}Once upon a time",
    "only special token": EOT,
    "non-breaking space": "a b c",
    "mixed scripts": "Привет мир, γειά σου κόσμε, مرحبا",
    "control characters": "a\x00b\x01c\x7f",
}


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    # the tests need the same tokenizer file the cases were made with
    shutil.copyfile(MODEL_ROOT / "data" / "tokenizer.json", FIXTURES / "tokenizer.json")
    tok = BPETokenizer.load(FIXTURES / "tokenizer.json")

    cases = []
    for name, text in CASES.items():
        ids = tok.encode(text)
        assert tok.decode(ids) == text, name
        cases.append({"name": name, "text": text, "ids": ids})

    out = {"vocab_size": tok.vocab_size, "cases": cases}
    (FIXTURES / "tokenizer_cases.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(cases)} cases")


if __name__ == "__main__":
    main()
