from __future__ import annotations

"""Tests for the tiktoken ranks adapter (Phase 2 compatibility).

Includes a real parity test against tiktoken's cl100k_base when the package
(and its downloaded BPE file) is available, plus self-contained synthetic
rank-file tests that always run.
"""

import base64
import os
import tempfile
import unittest
from typing import Any

from uniqtoken.tiktoken_adapter import (
    TiktokenEncoding,
    load_tiktoken_ranks,
)


def _write_synthetic_ranks() -> str:
    """Builds a tiny but valid .tiktoken file: all 256 bytes + a few merges."""
    lines = []
    for b in range(256):
        lines.append(base64.b64encode(bytes([b])) + b" " + str(b).encode())
    # merges (rank order matters): "th", "e ", " the", "the"
    for token, rank in [(b"th", 256), (b"e ", 257), (b" the", 258), (b"the", 259)]:
        lines.append(base64.b64encode(token) + b" " + str(rank).encode())
    fd, path = tempfile.mkstemp(suffix=".tiktoken")
    with os.fdopen(fd, "wb") as f:
        f.write(b"\n".join(lines) + b"\n")
    return path


SYNTHETIC_SPECIALS = {"<|endoftext|>": 1000, "<|fim|>": 1001}


class SyntheticRanksTests(unittest.TestCase):
    ranks_path: str
    enc: Any

    @classmethod
    def setUpClass(cls):
        cls.ranks_path = _write_synthetic_ranks()
        cls.enc = TiktokenEncoding.from_file(
            cls.ranks_path, name="synthetic", pattern="gpt2", special_tokens=SYNTHETIC_SPECIALS
        )

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.ranks_path)

    def test_load_ranks(self):
        ranks = load_tiktoken_ranks(self.ranks_path)
        self.assertEqual(len(ranks), 260)
        self.assertEqual(ranks[b"th"], 256)
        self.assertEqual(ranks[b"\x00"], 0)

    def test_greedy_merge_semantics(self):
        # b"the": pair (t,h) rank 256 merges first, then b"the" rank 259
        self.assertEqual(self.enc.encode("the"), [259])
        # b" the" exists directly (258) — whole piece gets its rank
        self.assertEqual(self.enc.encode(" the"), [258])

    def test_roundtrip_unicode_and_emoji(self):
        text = "héllo wörld — 你好 🌍 👨‍👩‍👧‍👦"
        ids = self.enc.encode(text)
        self.assertEqual(self.enc.decode(ids), text)
        for tid in ids:
            self.assertLess(tid, 256, "unseen text must fall back to single bytes in this vocab")

    def test_special_tokens(self):
        text = "hello <|endoftext|> world"
        # Disallowed special tokens now raise ValueError, matching tiktoken's exact behavior.
        with self.assertRaises(ValueError):
            self.enc.encode(text)
        ids = self.enc.encode(text, allowed_special="all")
        self.assertIn(1000, ids)
        self.assertEqual(self.enc.decode(ids), text)
        # With <|fim|> allowed but <|endoftext|> not, encoding must raise
        # because the text contains the disallowed <|endoftext|>.
        with self.assertRaises(ValueError):
            self.enc.encode(text, allowed_special={"<|fim|>"})
        # The two leading/trailing chunks around the disallowed special must still be byte-encodable
        # when the special itself is allowed but other specials are not.
        chunked = self.enc.encode("hello <|endoftext|> world", allowed_special={"<|endoftext|>"})
        self.assertIn(1000, chunked)

    def test_decode_rejects_unknown_ids(self):
        with self.assertRaises(ValueError):
            self.enc.decode([99999])

    def test_to_uniqtoken_bpe_model_ids_preserved(self):
        model = self.enc.to_uniqtoken_bpe_model()
        self.assertEqual(model.token_to_id["the"], 259)
        self.assertEqual(model.token_to_id[" the"], 258)
        self.assertEqual(model.id_to_token[256], "th")
        # UniqToken's BPE merge semantics on a single word must match tiktoken's
        # per-piece result for pieces without the pattern's split boundaries.
        self.assertEqual(model._encode_word("the"), ["the"])
        self.assertEqual(model._encode_word(" th"), [" ", "th"])

    def test_malformed_file_rejected(self):
        fd, path = tempfile.mkstemp(suffix=".tiktoken")
        with os.fdopen(fd, "wb") as f:
            f.write(b"not-base64-and-no-rank\n")
        try:
            with self.assertRaises(ValueError):
                load_tiktoken_ranks(path)
        finally:
            os.unlink(path)


class RealTiktokenParityTests(unittest.TestCase):
    """Differential tests against the real tiktoken cl100k_base encoding.

    Skipped automatically when tiktoken isn't installed, the ranks aren't
    cached locally, or the download fails (offline CI).
    """

    ref: Any
    ranks_path: str
    specials: Any

    @classmethod
    def setUpClass(cls):
        try:
            import tiktoken
        except ImportError:
            raise unittest.SkipTest("tiktoken package not installed")
        try:
            cls.ref = tiktoken.get_encoding("cl100k_base")
        except Exception:
            raise unittest.SkipTest("cl100k_base ranks not available (offline?)")
        ranks = getattr(cls.ref, "_mergeable_ranks", None)
        if not ranks:
            raise unittest.SkipTest("cannot access mergeable ranks on this tiktoken version")
        fd, cls.ranks_path = tempfile.mkstemp(suffix=".tiktoken")
        with os.fdopen(fd, "wb") as f:
            for token_bytes, rank in ranks.items():
                f.write(base64.b64encode(token_bytes) + b" " + str(rank).encode() + b"\n")
        cls.specials = dict(cls.ref._special_tokens)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.ranks_path)

    def _build_adapter(self) -> TiktokenEncoding:
        return TiktokenEncoding.from_file(
            self.ranks_path,
            name="cl100k_base",
            pattern="cl100k_base",
            special_tokens=self.specials,
            explicit_n_vocab=self.ref.n_vocab,
        )

    def test_exact_id_parity_with_tiktoken(self):
        enc = self._build_adapter()
        samples = [
            ("Hello, world!", set()),
            ("def foo(bar: int) -> int:\n    return bar + 42  # comment", set()),
            ("Emoji 👨‍👩‍👧‍👦 and Arabic مرحبا and CJK 中文 and combining é", set()),
            ("numbers 1234567 and 3.14159, tabs\tand\nnewlines\r\n", set()),
            ("hello <|endoftext|> world", {"<|endoftext|>"}),
            ("a" * 300, set()),
            ("", set()),
        ]
        for text, allowed in samples:
            ref_ids = self.ref.encode(text, allowed_special=allowed)
            ours = enc.encode(text, allowed_special=allowed)
            self.assertEqual(ours, ref_ids, f"ID mismatch for {text[:40]!r}")
            self.assertEqual(enc.decode(ours), self.ref.decode(ref_ids))


if __name__ == "__main__":
    unittest.main()
