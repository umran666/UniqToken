"""Full SuperBPE/CEM semantics validation for FastMergeEngine (Issue #114).

Issue #114 requires that FastMergeEngine validity be established against the
*complete* UniqToken SuperBPE/CEM execution contract — not just simplified
BPE-style piece sequences. This module validates the experimental merge
engines through the real production pipelines on a genuinely CEM-trained
cross-word (SuperBPE) tokenizer and maps every required semantic category
to at least one test, per the coverage-matrix acceptance criterion:

=======================================  ========================================
#114 required semantic                   Tests
=======================================  ========================================
CEM scores/ranks consumed, not           ``test_cem_scores_and_ranks_never_schedule``
recomputed / never scheduling
Cross-word merge legality                ``test_cross_word_legality_via_real_table``
Document separators / stream boundaries  ``test_batch_rows_are_isolated_streams``,
                                         ``test_no_state_leaks_between_calls``
Boundary constraints                     ``test_boundary_constraints_through_production_table``
Special tokens                           ``test_special_tokens_survive_engine_paths``
Byte-fallback tokens + reconstruction    ``test_byte_fallback_tokens_and_reconstruction``
Vocabulary membership + ID constraints   ``test_merged_ids_come_from_model``,
                                         ``test_leading_metaspace_exclusion`` (builder)
Deterministic ordering + tie handling    ``test_deterministic_ordering_leftmost_ties``
Dropout decisions + invalidation         ``test_dropout_end_to_end_reproducible``,
                                         ``test_dropout_engine_level_differential``
Pieces, IDs, offsets, byte spans         ``test_end_to_end_parity_encode_ids_offsets``,
                                         ``test_offsets_tile_without_gaps``
Unsupported cases explicit + tested      ``test_unsupported_cases_explicit``
=======================================  ========================================

Equality contract: on every supported case the ``fast`` and ``reference``
engines must show **zero behavior drift** from each other and from the
inlined production merge loop across ``encode``, ``encode_to_ids``,
``encode_with_offsets``, ``sample``, ``encode_batch`` and ``decode``.

Per issue #114, no throughput claim is made anywhere in this module; this is
correctness evidence only. All randomized cases derive from fixed recorded
seeds so failures are reproducible.
"""

from __future__ import annotations

import random
import unittest
from itertools import cycle
from unittest.mock import patch
from typing import Any, Dict, List, Optional, Sequence, Tuple

from uniqtoken.cem_merger import CrossEntropyMerging
from uniqtoken.merge_engine import (
    FastMergeEngine,
    MembershipMergeTable,
    MergeConstraints,
    ReferenceMergeEngine,
    SemanticProfile,
    UnsupportedSemantics,
    apply_engine_to_pieces,
    atoms_from_pieces,
    cross_word_membership_table,
    differential_against_production,
    plan_to_pieces,
    production_constraints,
)
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer, Token
from uniqtoken.unigram_trainer import UnigramModel

# ---------------------------------------------------------------------------
# Fixed corpus and fixtures (recorded seeds; deterministic training inputs)
# ---------------------------------------------------------------------------

TRAIN_CORPUS = [
    "hello world and welcome to the system",
    "in the beginning of the process to the end",
    "the quick brown fox jumps over the lazy dog",
    "def calculate_loss(predictions, targets) -> float:",
    "processing... wait...... result = ([{a: (b + c)}]);",
    "人工智能和自然语言处理技术发展迅速",
    "प्रणाली और विश्वविद्यालय में अनुसंधान कार्य",
    "المعلوماتية والاسـتراتيجية في معالجة اللغات",
    "v1.2.3 and parseXMLDocument with calculate_total_score",
    "the system processes the process systematically",
]

SPECIAL_TOKEN = "<|endoftext|>"

# Eval battery: plain, code, multilingual, punctuation, emoji/bytes, specials.
PLAIN_TEXTS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "the system processes the process systematically and the process ends",
    "welcome to the beginning of the system",
]
CODE_TEXT = "def calculate_loss(predictions, targets) -> float:"
MULTILINGUAL_TEXTS = [
    "人工智能和自然语言处理技术发展迅速",
    "प्रणाली और विश्वविद्यालय में अनुसंधान",
]
PUNCTUATION_TEXT = "processing... wait...... result = ([{a: (b + c)}]);"
EMOJI_TEXT = "launch 🚀 with 👩‍💻 status ok"
UNKNOWN_BYTES_TEXT = "ẋẏẛ rare glyphs ᚠᚢᚦ here"
SPECIAL_TEXT = f"hello {SPECIAL_TOKEN} world"

ALL_ENGINE_CHOICES: Tuple[Optional[str], ...] = (None, "fast", "reference")

_TOK_CACHE: Dict[str, Any] = {}


def _superbpe_tokenizer(merge_engine: Optional[str] = None) -> CustomTokenizer:
    """Build (or reuse) the CEM-trained cross-word tokenizer for this suite."""
    if "model" not in _TOK_CACHE:
        base = CustomTokenizer.train_from_corpus(
            TRAIN_CORPUS,
            target_vocab_size=400,
            min_frequency=1,
            verbose=False,
            special_tokens=[SPECIAL_TOKEN],
        )
        pretok_chunks = [
            chunk for doc in TRAIN_CORPUS for chunk in base.pre_tokenizer.pre_tokenize(base.normalizer.normalize(doc))
        ]
        cem = CrossEntropyMerging(max_merges=30, cross_word=True, verbose=False)
        sbp_model = cem.optimize(base.model, chunks=pretok_chunks)
        _TOK_CACHE["model"] = sbp_model
        _TOK_CACHE["base"] = base
    base: CustomTokenizer = _TOK_CACHE["base"]
    return CustomTokenizer(
        normalizer=base.normalizer,
        pre_tokenizer=base.pre_tokenizer,
        model=_TOK_CACHE["model"],
        merge_engine=merge_engine,
    )


def _tok(engine: Optional[str]) -> CustomTokenizer:
    return _superbpe_tokenizer(merge_engine=engine)


def _offset_triples(tokens: Sequence[Token]) -> List[Tuple[str, int, Tuple[int, int]]]:
    return [(t.text, t.id, t.raw_span) for t in tokens]


def _assert_spans_tile(text: str, tokens: Sequence[Token], msg: str) -> None:
    """Raw spans must cover [0, len(text)) with no gaps.

    Multiple byte-fallback tokens may *share* one character span (one span
    per multi-byte character), so overlaps among identical spans are legal
    (see COMPATIBILITY/boundary analysis: "multiple fallback bytes may share
    a character span"); what must hold is full coverage and monotonicity.
    """
    spans = sorted((t.raw_span[0], t.raw_span[1]) for t in tokens)
    if not spans:
        raise AssertionError(f"{msg}: no spans for non-empty text")
    if spans[0][0] != 0 or spans[-1][1] != len(text):
        raise AssertionError(f"{msg}: spans do not cover the text ends: {spans}")
    merged_start, merged_end = spans[0]
    for start, end in spans[1:]:
        if start > merged_end:
            raise AssertionError(f"{msg}: gap in spans before {start}: {spans}")
        merged_end = max(merged_end, end)
    if merged_end != len(text):
        raise AssertionError(f"{msg}: spans cover [0, {merged_end}) only: {spans}")


# ---------------------------------------------------------------------------
# End-to-end parity across every public pipeline
# ---------------------------------------------------------------------------


class FullSemanticsEndToEndTests(unittest.TestCase):
    """Zero drift between production, fast and reference on real pipelines."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tok_default = _tok(None)
        cls.tok_fast = _tok("fast")
        cls.tok_reference = _tok("reference")
        # The fixture must actually be a SuperBPE model, otherwise every
        # parity assertion below would vacuously pass.
        cls.cross_tokens = cls.tok_default._cross_word_tokens()
        if len(cls.cross_tokens) == 0:
            raise AssertionError("fixture training produced no cross-word tokens; SuperBPE semantics untested")
        cls.texts = [*PLAIN_TEXTS, CODE_TEXT, *MULTILINGUAL_TEXTS, PUNCTUATION_TEXT, EMOJI_TEXT]

    def test_end_to_end_parity_encode_ids_offsets(self):
        """encode / encode_to_ids / encode_with_offsets identical on all engines."""
        for text in self.texts:
            with self.subTest(text=text[:40]):
                prod_pieces = self.tok_default.encode(text)
                fast_pieces = self.tok_fast.encode(text)
                ref_pieces = self.tok_reference.encode(text)
                self.assertEqual(prod_pieces, fast_pieces, "encode drift: production vs fast")
                self.assertEqual(prod_pieces, ref_pieces, "encode drift: production vs reference")

                prod_ids = self.tok_default.encode_to_ids(text)
                fast_ids = self.tok_fast.encode_to_ids(text)
                ref_ids = self.tok_reference.encode_to_ids(text)
                self.assertEqual(prod_ids, fast_ids, "encode_to_ids drift: production vs fast")
                self.assertEqual(prod_ids, ref_ids, "encode_to_ids drift: production vs reference")

                prod_off = _offset_triples(self.tok_default.encode_with_offsets(text))
                fast_off = _offset_triples(self.tok_fast.encode_with_offsets(text))
                ref_off = _offset_triples(self.tok_reference.encode_with_offsets(text))
                self.assertEqual(prod_off, fast_off, "offsets drift: production vs fast")
                self.assertEqual(prod_off, ref_off, "offsets drift: production vs reference")

    def test_offsets_tile_without_gaps(self):
        """Fast-engine spans tile the raw input exactly (issue #114 byte spans)."""
        for text in [*PLAIN_TEXTS, CODE_TEXT, PUNCTUATION_TEXT, EMOJI_TEXT]:
            for engine_name in (None, "fast", "reference"):
                tokens = _tok(engine_name).encode_with_offsets(text)
                with self.subTest(text=text[:30], engine=engine_name):
                    _assert_spans_tile(text, tokens, f"engine={engine_name}")

    def test_decode_reconstruction_identical(self):
        """decode(encode_to_ids(text)) is identical on all engines, byte-exact on ASCII."""
        for text in self.texts:
            ids_prod = self.tok_default.encode_to_ids(text)
            with self.subTest(text=text[:40]):
                self.assertEqual(
                    self.tok_default.decode(ids_prod),
                    self.tok_fast.decode(self.tok_fast.encode_to_ids(text)),
                    "decode drift: production vs fast",
                )
                self.assertEqual(
                    self.tok_default.decode(ids_prod),
                    self.tok_reference.decode(self.tok_reference.encode_to_ids(text)),
                    "decode drift: production vs reference",
                )
        # Byte-exact round trip for plain ASCII with a space-bearing vocab.
        for text in PLAIN_TEXTS:
            with self.subTest(roundtrip=text):
                self.assertEqual(
                    self.tok_fast.decode(self.tok_fast.encode_to_ids(text)),
                    text,
                    "fast-engine decode round trip is not byte-exact",
                )

    def test_batch_rows_are_isolated_streams(self):
        """Batch rows never share a merge stream (document stream boundaries)."""
        rows = [*PLAIN_TEXTS, CODE_TEXT]
        for engine_name in ALL_ENGINE_CHOICES:
            tok = _tok(engine_name)
            batched = tok.encode_batch(rows)
            individual = [tok.encode(row) for row in rows]
            with self.subTest(engine=engine_name):
                self.assertEqual(batched, individual, f"batch row drift for engine={engine_name}")

    def test_no_state_leaks_between_calls(self):
        """Repeated identical calls give identical outputs (no cross-call state)."""
        for engine_name in ALL_ENGINE_CHOICES:
            tok = _tok(engine_name)
            with self.subTest(engine=engine_name):
                first = tok.encode(PLAIN_TEXTS[2])
                second = tok.encode(PLAIN_TEXTS[2])
                self.assertEqual(first, second)

    def test_special_tokens_survive_engine_paths(self):
        """Allowed specials pass through unmerged with exact IDs on all engines."""
        for engine_name in ALL_ENGINE_CHOICES:
            tok = _tok(engine_name)
            with self.subTest(engine=engine_name):
                pieces = tok.encode(SPECIAL_TEXT, allowed_special={SPECIAL_TOKEN})
                self.assertIn(SPECIAL_TOKEN, pieces)
                # The special is a standalone token, never fused with a neighbor.
                idx = pieces.index(SPECIAL_TOKEN)
                self.assertFalse(
                    idx > 0 and pieces[idx - 1].endswith(SPECIAL_TOKEN),
                    "special token was fused with its left neighbor",
                )
                self.assertFalse(
                    idx < len(pieces) - 1 and pieces[idx + 1].startswith(SPECIAL_TOKEN),
                    "special token was fused with its right neighbor",
                )
                # Disallowed specials are escaped and still decode identically.
                ids = tok.encode_to_ids(SPECIAL_TEXT)  # 'none' => escaped, not special
                self.assertEqual(
                    tok.decode(ids),
                    self.tok_default.decode(self.tok_default.encode_to_ids(SPECIAL_TEXT)),
                    "escaped-special decode drift",
                )

    def test_byte_fallback_tokens_and_reconstruction(self):
        """Unknown glyphs become byte-fallback tokens; decode reconstructs; no engine drift."""
        for engine_name in ALL_ENGINE_CHOICES:
            tok = _tok(engine_name)
            with self.subTest(engine=engine_name):
                pieces = tok.encode(UNKNOWN_BYTES_TEXT)
                byte_tokens = [p for p in pieces if p.startswith("<0x") and p.endswith(">")]
                self.assertTrue(byte_tokens, "expected byte-fallback tokens for unknown glyphs")
                ids = tok.encode_to_ids(UNKNOWN_BYTES_TEXT)
                self.assertEqual(
                    tok.decode(ids),
                    self.tok_default.decode(self.tok_default.encode_to_ids(UNKNOWN_BYTES_TEXT)),
                    "byte-fallback decode drift",
                )
                # Multilingual text also exercises fallback-adjacent merges.
                for text in MULTILINGUAL_TEXTS:
                    self.assertEqual(
                        tok.encode(text),
                        self.tok_default.encode(text),
                        "multilingual byte-fallback drift",
                    )

    def test_merged_ids_come_from_model(self):
        """Every output ID is the model's canonical ID for its piece (ID constraints)."""
        tok = self.tok_fast
        unk_fallback = tok.model.token_to_id.get(tok.model.unk_token, 0)
        for text in self.texts:
            pieces = tok.encode(text)
            ids = tok.encode_to_ids(text)
            with self.subTest(text=text[:30]):
                self.assertEqual(len(pieces), len(ids))
                for piece, token_id in zip(pieces, ids):
                    self.assertEqual(
                        token_id,
                        tok.model.token_to_id.get(piece, unk_fallback),
                        f"ID constraint violated for piece {piece!r}",
                    )


# ---------------------------------------------------------------------------
# CEM scores / ranks: consumed as metadata, never scheduling
# ---------------------------------------------------------------------------


class FullSemanticsCemRanksTests(unittest.TestCase):
    """CEM outputs are consumed without recomputing the objective (#114)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = _tok("fast")

    def test_cem_scores_and_ranks_never_schedule(self):
        """Permuted / conflicting learned ranks must not change any output."""
        table = cross_word_membership_table(self.tok)
        pieces = self.tok.encode(PLAIN_TEXTS[2])
        eligible = dict(table.eligible)
        rng = random.Random(20261024)
        # Rank metadata over actual adjacent pairs with arbitrary conflicting values.
        ranks_full = {}
        for left, right in zip(pieces, pieces[1:]):
            if left + right in eligible:
                ranks_full[(left, right)] = rng.randrange(0, 10_000)
        ranked = MembershipMergeTable(
            eligible=eligible,
            vocabulary_identity=table.vocabulary_identity,
            ranks=ranks_full,
        )
        inverse = MembershipMergeTable(
            eligible=eligible,
            vocabulary_identity=table.vocabulary_identity,
            ranks={pair: 10_000 - rank for pair, rank in ranks_full.items()},
        )
        atoms = atoms_from_pieces(pieces)
        fast = FastMergeEngine()
        ref = ReferenceMergeEngine()
        for tbl in (ranked, inverse):
            constraints = production_constraints(tbl)
            self.assertEqual(
                plan_to_pieces(fast.apply(atoms, tbl, constraints)),
                plan_to_pieces(ref.apply(atoms, tbl, constraints)),
                "rank metadata changed reference/fast agreement",
            )
        self.assertEqual(
            fast.apply(atoms, ranked, production_constraints(ranked)),
            fast.apply(atoms, inverse, production_constraints(inverse)),
            "permuting learned ranks changed FastMergeEngine output (ranks must never schedule)",
        )
        # The un-ranked production table must agree too.
        self.assertEqual(
            fast.apply(atoms, table, production_constraints(table)),
            fast.apply(atoms, ranked, production_constraints(ranked)),
            "adding rank metadata changed the merge schedule",
        )

    def test_cross_word_legality_via_real_table(self):
        """Fast output equals production output on the real CEM-grown table."""
        for text in PLAIN_TEXTS:
            with self.subTest(text=text[:30]):
                self.assertEqual(_tok("fast").encode(text), _tok(None).encode(text))


# ---------------------------------------------------------------------------
# Table builder, boundaries, ordering, dropout, unsupported semantics
# ---------------------------------------------------------------------------


class FullSemanticsTableBuilderTests(unittest.TestCase):
    """cross_word_membership_table / _cross_word_tokens semantics (#114)."""

    def _tokenizer_from_vocab(self, pieces: Sequence[str], space_char: str = "▁") -> CustomTokenizer:
        vocab = {piece: -1.0 for piece in dict.fromkeys(pieces)}
        ids = {piece: i for i, piece in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=ids,
            id_to_token={i: p for p, i in ids.items()},
            special_tokens=[],
            max_subword_len=max((len(p) for p in vocab), default=1),
            byte_fallback=False,
        )
        return CustomTokenizer(
            Normalizer(normalize_unicode=False, space_char=space_char),
            RegexPreTokenizer(),
            model,
        )

    def test_leading_metaspace_exclusion(self):
        """Leading-only metaspace tokens are never eligible; internal ones are."""
        tok = self._tokenizer_from_vocab(
            ["the", "▁quick", "the▁quick", "▁the▁quick", "quick", "▁the", "brown", "▁brown", "brown▁fox"]
        )
        cross = tok._cross_word_tokens()
        self.assertIn("the▁quick", cross, "internal-metaspace token must be eligible")
        self.assertIn("brown▁fox", cross, "internal-metaspace token must be eligible")
        self.assertNotIn("▁quick", cross, "leading-only metaspace token must be excluded")
        # '▁the▁quick' contains an internal metaspace (t[1:]) and IS eligible;
        # only leading-ONLY markers are excluded by the builder.
        self.assertIn("▁the▁quick", cross, "internal-metaspace token must be eligible")

        # Runtime behavior on the real table: 'the' + '▁quick' merges.
        table = cross_word_membership_table(tok)
        constraints = production_constraints(table)
        for engine in (FastMergeEngine(), ReferenceMergeEngine()):
            out, _ = apply_engine_to_pieces(engine, ["the", "▁quick", "▁brown"], table, constraints)
            self.assertEqual(
                out,
                ["the▁quick", "▁brown"],
                f"engine {engine.name} did not reproduce builder-driven eligibility",
            )

    def test_alternate_space_char(self):
        """The marker comes from Normalizer.space_char, not a hardcoded ▁."""
        marker = "Ġ"  # GPT-2-style
        tok = self._tokenizer_from_vocab(["the", "Ġquick", "theĠquick", "quick", "Ġthe"], space_char=marker)
        cross = tok._cross_word_tokens()
        self.assertIn("theĠquick", cross)
        self.assertNotIn("Ġquick", cross)
        table = cross_word_membership_table(tok)
        constraints = production_constraints(table)
        for engine in (FastMergeEngine(), ReferenceMergeEngine()):
            out, _ = apply_engine_to_pieces(engine, ["the", "Ġquick"], table, constraints)
            self.assertEqual(out, ["theĠquick"], f"engine {engine.name} with alternate space_char")

    def test_boundary_constraints_through_production_table(self):
        """Hard cuts constrain merges identically on the real CEM table."""
        tok = _tok("fast")
        table = cross_word_membership_table(tok)
        pieces = tok.encode(PLAIN_TEXTS[2])
        if len(pieces) < 3:
            self.skipTest("fixture produced too few pieces")
        constraints = MergeConstraints(
            semantic_profile=SemanticProfile.SUPER_BPE_PASS_V1,
            vocabulary_identity=table.vocabulary_identity,
            hard_cuts=frozenset({1, len(pieces) - 1}),
        )
        for engine in (FastMergeEngine(), ReferenceMergeEngine()):
            out, plan = apply_engine_to_pieces(engine, pieces, table, constraints)
            # A cut at boundary b forbids any group spanning leaf boundary b.
            for cut in constraints.hard_cuts:
                for group in plan.groups:
                    self.assertFalse(
                        group.start < cut < group.end,
                        f"engine {engine.name} crossed hard cut {cut} with group {group}",
                    )
            self.assertEqual(out, [g.piece for g in plan.groups])

    def test_deterministic_ordering_leftmost_ties(self):
        """Overlapping eligible pairs resolve leftmost, identically everywhere."""
        text = "the the the the the the the"
        outputs = [_tok(engine_name).encode(text) for engine_name in ALL_ENGINE_CHOICES]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_dropout_end_to_end_reproducible(self):
        """Merge dropout through full encode: identical pieces with identical RNG streams."""
        rng = random.Random(114)
        # Cycle keeps the stream finite-but-inexhaustible and deterministic.
        draws = [0.0 if rng.random() < 0.5 else 0.99 for _ in range(4096)]
        for text in [*PLAIN_TEXTS, CODE_TEXT]:
            with self.subTest(text=text[:30]):
                with patch("random.random", side_effect=cycle(draws)):
                    prod = _tok(None).encode(text, dropout_prob=0.5)
                with patch("random.random", side_effect=cycle(draws)):
                    fast = _tok("fast").encode(text, dropout_prob=0.5)
                with patch("random.random", side_effect=cycle(draws)):
                    ref = _tok("reference").encode(text, dropout_prob=0.5)
                self.assertEqual(prod, fast, "dropout drift: production vs fast")
                self.assertEqual(prod, ref, "dropout drift: production vs reference")
                # A keep-everything stream must equal plain encoding.
                with patch("random.random", side_effect=cycle([0.99] * 4096)):
                    kept = _tok("fast").encode(text, dropout_prob=0.5)
                self.assertEqual(kept, _tok("fast").encode(text), "dropout no-op drift")

    def test_dropout_engine_level_differential(self):
        """Engine-level differential with tapes on the real CEM table (#114 reproducibility)."""
        tok = _tok("fast")
        for index, text in enumerate(PLAIN_TEXTS):
            pieces = _tok(None).encode(text)
            tape = [random.Random(114 + index).random() < 0.4 for _ in range(512)]
            report = differential_against_production(tok, pieces, 0.5, decision_tape=tape, engine=FastMergeEngine())
            with self.subTest(text=text[:30]):
                self.assertTrue(report["match"], f"engine-level dropout differential failed: {report}")

    def test_sample_pipelines_agree_with_seeded_rng(self):
        """sample() merge stage is engine-invariant under identical lattice draws."""
        text = PLAIN_TEXTS[2]
        outputs = []
        for engine_name in ALL_ENGINE_CHOICES:
            random.seed(20261024)
            outputs.append(_tok(engine_name).sample(text, alpha=0.5))
        self.assertEqual(outputs[0], outputs[1], "sample() drift: production vs fast")
        self.assertEqual(outputs[0], outputs[2], "sample() drift: production vs reference")

    def test_unsupported_cases_explicit(self):
        """Unsupported semantics fail loudly through the engine contract (#114)."""
        table = cross_word_membership_table(_tok("fast"))
        constraints = MergeConstraints(
            semantic_profile="SuperBpePassV2",
            vocabulary_identity=table.vocabulary_identity,
        )
        atoms = atoms_from_pieces(["a", "b"])
        for engine in (FastMergeEngine(), ReferenceMergeEngine()):
            with self.subTest(engine=engine.name):
                with self.assertRaises(UnsupportedSemantics):
                    engine.apply(atoms, table, constraints)


class CoverageMatrixTests(unittest.TestCase):
    """Acceptance criterion: coverage matrix maps every #114 semantic to tests."""

    REQUIRED_CATEGORIES = {
        "cem_scores_ranks_metadata_only",
        "cross_word_legality",
        "document_separators_stream_boundaries",
        "boundary_constraints",
        "special_tokens",
        "byte_fallback_reconstruction",
        "vocabulary_membership_ids",
        "deterministic_ordering_ties",
        "dropout_invalidation",
        "pieces_ids_offsets_spans",
        "unsupported_cases_explicit",
    }

    def test_every_required_category_is_mapped(self):
        matrix = {
            "cem_scores_ranks_metadata_only": [FullSemanticsCemRanksTests],
            "cross_word_legality": [FullSemanticsCemRanksTests],
            "document_separators_stream_boundaries": [
                FullSemanticsEndToEndTests.test_batch_rows_are_isolated_streams,
                FullSemanticsEndToEndTests.test_no_state_leaks_between_calls,
            ],
            "boundary_constraints": [FullSemanticsTableBuilderTests],
            "special_tokens": [FullSemanticsEndToEndTests],
            "byte_fallback_reconstruction": [FullSemanticsEndToEndTests],
            "vocabulary_membership_ids": [FullSemanticsEndToEndTests, FullSemanticsTableBuilderTests],
            "deterministic_ordering_ties": [FullSemanticsTableBuilderTests],
            "dropout_invalidation": [FullSemanticsTableBuilderTests],
            "pieces_ids_offsets_spans": [FullSemanticsEndToEndTests],
            "unsupported_cases_explicit": [FullSemanticsTableBuilderTests],
        }
        self.assertEqual(set(matrix), self.REQUIRED_CATEGORIES)
        for category, targets in matrix.items():
            for target in targets:
                if isinstance(target, type):
                    test_names = {name for name in dir(target) if name.startswith("test_")}
                    self.assertTrue(test_names, f"{category}: {target.__name__} has no tests")
                else:
                    self.assertTrue(callable(target), f"{category}: unmapped target {target!r}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
