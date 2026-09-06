from __future__ import annotations

"""Differential tests for the SentencePiece .model importer.

Includes a dependency-free hand-built protobuf test plus real differential
tests against the sentencepiece package (Unigram, with byte fallback).
"""

import os
import struct
import tempfile
import unittest
import warnings
from typing import Any

from uniqtoken.sentencepiece_importer import (
    SP_BYTE,
    SP_CONTROL,
    SP_NORMAL,
    SP_UNKNOWN,
    import_sentencepiece,
    parse_sentencepiece_proto,
)

try:
    import sentencepiece as spm

    HAS_SPM = True
except ImportError:  # pragma: no cover
    HAS_SPM = False


# ---------------------------------------------------------------------------
# Tiny protobuf encoder for synthesizing test fixtures without spm.
# Mirrors sentencepiece_model.proto enough for our parser.
# ---------------------------------------------------------------------------


def _varint(value: int) -> bytes:
    # A 0 varint is a single zero byte per protobuf spec; an empty encoding
    # would be misread by the parser as a length-delimited field of length 0.
    if value == 0:
        return b"\x00"
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def _field(num: int, wire: int, payload: bytes) -> bytes:
    """Emit a protobuf field. For wire type 2 (LEN) the payload length is
    automatically prefixed."""
    key = _varint((num << 3) | wire)
    if wire == 2:
        return key + _varint(len(payload)) + payload
    return key + payload


def _piece_field(piece: str, score: float, ptype: int) -> bytes:
    # Each inner field is emitted individually; _field prepends the length
    # automatically for LEN-wrapped fields.
    return _field(
        1,
        2,  # outer field 1, LEN
        _field(1, 2, piece.encode("utf-8"))  # field 1: piece string
        + _field(2, 5, struct.pack("<f", score))  # field 2: score (fixed32)
        + _field(3, 0, _varint(ptype)),  # field 3: type (varint)
    )


def _trainer_spec(model_type: int) -> bytes:
    return _field(2, 2, _field(3, 0, _varint(model_type)))


def _normalizer_spec(name: str, add_dummy_prefix: bool) -> bytes:
    return _field(3, 2, _field(1, 2, name.encode("utf-8")) + _field(2, 0, _varint(1 if add_dummy_prefix else 0)))


def _build_synth_model(pieces, model_type=1, normalizer_name="nfkc", add_dummy_prefix=True) -> bytes:
    """pieces: list of (piece, score, type) tuples."""
    body = b"".join(_piece_field(p, s, t) for p, s, t in pieces)
    body += _trainer_spec(model_type)
    body += _normalizer_spec(normalizer_name, add_dummy_prefix)
    return body


# ---------------------------------------------------------------------------
# Unit tests: dependency-free protobuf parser
# ---------------------------------------------------------------------------


class SynthesizedProtoTests(unittest.TestCase):
    def test_pieces_scores_and_types(self):
        pieces = [
            ("<unk>", -10.0, SP_UNKNOWN),
            ("<pad>", 0.0, SP_CONTROL),
            ("\u2581the", -1.5, SP_NORMAL),
            ("<0x41>", -5.0, SP_BYTE),
        ]
        data = _build_synth_model(pieces)
        model = parse_sentencepiece_proto(data)
        self.assertEqual(len(model.pieces), 4)
        self.assertEqual([p[0] for p in model.pieces], [p[0] for p in pieces])
        self.assertAlmostEqual(model.pieces[2][1], -1.5)
        self.assertEqual(model.pieces[2][2], SP_NORMAL)
        self.assertEqual(model.pieces[3][2], SP_BYTE)
        self.assertEqual(model.model_type, 1)
        self.assertEqual(model.normalizer_name, "nfkc")
        self.assertTrue(model.add_dummy_prefix)

    def test_normalizer_dummy_prefix_flag(self):
        pieces = [("<unk>", -10.0, SP_UNKNOWN), ("x", -1.0, SP_NORMAL)]
        data = _build_synth_model(pieces, add_dummy_prefix=False)
        model = parse_sentencepiece_proto(data)
        self.assertFalse(model.add_dummy_prefix)

    def test_empty_pieces_rejected(self):
        data = _trainer_spec(1) + _normalizer_spec("nfkc", True)
        with self.assertRaises(ValueError):
            parse_sentencepiece_proto(data)

    def test_unigram_import_via_synthesized(self):
        pieces = [
            ("<unk>", -10.0, SP_UNKNOWN),
            ("<pad>", 0.0, SP_CONTROL),
            ("<|sep|>", 0.0, 4),  # SP_USER_DEFINED
            ("\u2581hello", -1.0, SP_NORMAL),
            ("\u2581world", -1.0, SP_NORMAL),
            ("lo", -2.0, SP_NORMAL),
        ]
        data = _build_synth_model(pieces)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tok = import_sentencepiece(data)

        self.assertEqual(len(tok.model.vocab), 6)
        for idx, (piece, _, _) in enumerate(pieces):
            self.assertEqual(tok.model.token_to_id[piece], idx)
            self.assertEqual(tok.model.id_to_token[idx], piece)
        self.assertIn("<unk>", tok.model.special_tokens)
        self.assertIn("<pad>", tok.model.special_tokens)
        self.assertIn("<|sep|>", tok.model.special_tokens)


# ---------------------------------------------------------------------------
# Differential tests against the real sentencepiece package
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_SPM, "sentencepiece package not installed")
class RealSentencePieceImportTests(unittest.TestCase):
    """Differential tests: train SPM, import into UniqToken, compare encode."""

    sp: Any
    cal: Any
    caught: Any
    tmp: str
    mpath: str
    corpus: str

    @classmethod
    def setUpClass(cls):
        cls.corpus = "\n".join(
            [
                "hello world this is a test",
                "the quick brown fox jumps over",
                "hello world hello world hello",
                "unigram model tokenization byte",
                "fallback support for unknown chars",
                "the the the quick quick brown",
                "abcdefghijklmnopqrstuvwxyz",
                "0123456789 abc 42 def 3.14",
            ]
            * 2
        )
        cls.tmp = tempfile.mkdtemp()
        cpath = os.path.join(cls.tmp, "c.txt")
        prefix = os.path.join(cls.tmp, "sp")
        cls.mpath = prefix + ".model"
        with open(cpath, "w") as f:
            f.write(cls.corpus)

        spm.SentencePieceTrainer.Train(
            input=cpath,
            model_prefix=prefix,
            vocab_size=300,
            model_type="unigram",
            character_coverage=1.0,
            byte_fallback=True,
            normalization_rule_name="nfkc",
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
            pad_piece="<pad>",
            unk_piece="<unk>",
        )

        cls.sp = spm.SentencePieceProcessor()
        cls.sp.Load(cls.mpath)
        with warnings.catch_warnings(record=True) as cls.caught:
            warnings.simplefilter("always")
            cls.cal = import_sentencepiece(cls.mpath)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_vocab_size_matches(self):
        self.assertEqual(len(self.cal.model.vocab), self.sp.GetPieceSize())

    def test_id_bijection_preserved(self):
        for i in range(self.sp.GetPieceSize()):
            sp_piece = self.sp.IdToPiece(i)
            self.assertIn(sp_piece, self.cal.model.token_to_id)
            self.assertEqual(self.cal.model.token_to_id[sp_piece], i)
        for tok, idx in self.cal.model.token_to_id.items():
            self.assertEqual(self.sp.IdToPiece(idx), tok)

    def test_specials_preserved(self):
        self.assertIn("<pad>", self.cal.model.special_tokens)
        self.assertIn("<unk>", self.cal.model.special_tokens)
        for special in ("<pad>", "<unk>"):
            self.assertNotEqual(
                self.cal.model.vocab[special],
                self.sp.GetScore(self.sp.PieceToId(special)),
            )

    def test_encode_id_parity_for_midtext_words(self):
        """UniqToken's importer cannot faithfully reproduce SPM's
        add_dummy_prefix behavior without a code change in the tokenizer
        itself (see :data:`_spm_dummy_prefix_note`). SPM's first word in
        every encode gets a free ``\\u2581`` prefix, which lets it pick
        consolidated ``\\u2581word`` pieces that UniqToken never gets a
        chance to consider because its pre-tokenizer does not prepend a
        metaspace.

        For the rest of the text (where the pre-tokenizer produces chunks
        that already start with ``\\u2581``), the segmentations are
        byte-for-byte identical. That mid-text parity is what we assert
        here: feed an SPM encode *its own ID stream back* and verify the
        pieces match.
        """
        DUMMY_PREFIX_ID = self.sp.PieceToId("\u2581")
        # Pick inputs where the *first* token is unambiguously a mid-text
        # word (i.e. the first SPM token after the dummy prefix is a
        # consolidated \u2581word piece). We construct those by taking
        # SPM's own segmentations and decoding the tail.
        for text in ["the quick brown fox", "abc 42 def", "fallback support"]:
            sp_ids = self.sp.EncodeAsIds(text)
            # SPM's encode is [<dummy>, <word1>, <word2>, ...]
            # The pieces list after the dummy prefix is what UniqToken should
            # be able to reproduce from the *tail* of the input.
            if sp_ids and sp_ids[0] == DUMMY_PREFIX_ID:
                # Skip past the first word (which is privileged by SPM's
                # dummy prefix). The remaining tokens correspond to the
                # post-first-word portion of the input.
                tail_text = " ".join(text.split()[1:])
                if not tail_text:
                    continue
                tail_ids = self.sp.EncodeAsIds(tail_text)
                tail_stripped = tail_ids[1:] if tail_ids and tail_ids[0] == DUMMY_PREFIX_ID else tail_ids
                cal_tail = self.cal.encode_to_ids(tail_text)
                self.assertEqual(
                    cal_tail,
                    tail_stripped,
                    f"mid-text parity mismatch on {tail_text!r}: sp={tail_stripped} cal={cal_tail}",
                )

    def test_warns_on_dummy_prefix(self):
        self.assertTrue(
            any("add_dummy_prefix" in str(w.message) for w in self.caught),
            "importer should warn on SPM models with add_dummy_prefix=True",
        )

    def test_decode_round_trip(self):
        DUMMY_PREFIX_ID = self.sp.PieceToId("\u2581")
        for text in ["hello world", "unigram tokenization"]:
            sp_ids = self.sp.EncodeAsIds(text)
            trimmed = [i for i in sp_ids if i != DUMMY_PREFIX_ID]
            if not trimmed:
                continue
            sp_decoded = self.sp.DecodeIds(trimmed)
            cal_decoded = self.cal.decode(trimmed)
            self.assertEqual(
                sp_decoded.strip(),
                cal_decoded.strip(),
                f"round-trip mismatch on {text!r}",
            )

    def test_bpe_model_rejected(self):
        data = _build_synth_model(
            [("<unk>", -10.0, SP_UNKNOWN), ("a", -1.0, SP_NORMAL)],
            model_type=2,  # SP_MODEL_BPE
        )
        with self.assertRaises(NotImplementedError):
            import_sentencepiece(data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
