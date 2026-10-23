"""Tests for FastMergeEngine and MergeEngine integration (#115).

Verifies:
- Complete differential parity between ReferenceMergeEngine and FastMergeEngine
  across all review vectors and SuperBpePassV1 semantic constraints (#110, #114).
- Clean selection and fallback behind the generic MergeEngine contract (#115).
- Integration into CustomTokenizer via explicit argument and UNIQTOKEN_MERGE_ENGINE.
- Failure loudly on unsupported semantics / invalid configuration.
- Preservation of default tokenizer output and production engine selection.
- Save/load serialization compatibility.
"""

from __future__ import annotations

import os
import random
import tempfile
import unittest
from typing import Dict, List, Sequence
from unittest.mock import patch

from uniqtoken.merge_engine import (
    AlwaysAllowLegality,
    Atom,
    BooleanTapeDecisions,
    DecisionError,
    FastMergeEngine,
    InvalidConfiguration,
    InvariantViolation,
    MembershipMergeTable,
    MergeConstraints,
    MergeEngine,
    MergePlan,
    MergeRule,
    PredicateLegality,
    ReferenceMergeEngine,
    SemanticProfile,
    UnsupportedSemantics,
    apply_engine_to_pieces,
    apply_engine_to_tokens,
    atoms_from_pieces,
    cross_word_membership_table,
    differential_against_production,
    differential_reference_vs_fast,
    get_merge_engine,
    plan_to_pieces,
    plan_to_tokens,
    production_constraints,
)
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer, Token
from uniqtoken.unigram_trainer import UnigramModel

S = "\u2581"
A = "a"
B = S + "b"
C = S + "c"
D = S + "d"


def _model_from_pieces(pieces: Sequence[str], results: Sequence[str]) -> UnigramModel:
    vocab = {piece: -1.0 for piece in dict.fromkeys([*pieces, *results])}
    ids = {piece: i for i, piece in enumerate(vocab)}
    return UnigramModel(
        vocab=vocab,
        token_to_id=ids,
        id_to_token={i: piece for piece, i in ids.items()},
        special_tokens=[],
        max_subword_len=max((len(p) for p in vocab), default=1),
        byte_fallback=False,
    )


def _tokenizer(
    pieces: Sequence[str],
    results: Sequence[str],
    merge_engine: str | MergeEngine | None = None,
) -> CustomTokenizer:
    model = _model_from_pieces(pieces, results)
    return CustomTokenizer(
        Normalizer(normalize_unicode=False),
        RegexPreTokenizer(),
        model,
        merge_engine=merge_engine,
    )


def _table(results: Sequence[str], ids: Dict[str, int] | None = None) -> MembershipMergeTable:
    eligible = {r: (ids[r] if ids is not None else i) for i, r in enumerate(dict.fromkeys(results))}
    return MembershipMergeTable(eligible=eligible, vocabulary_identity=frozenset(eligible.items()))


def _constraints(table: MembershipMergeTable, **kwargs) -> MergeConstraints:
    return MergeConstraints(
        semantic_profile=SemanticProfile.SUPER_BPE_PASS_V1,
        vocabulary_identity=table.vocabulary_identity,
        hard_cuts=frozenset(kwargs.get("hard_cuts", ())),
        legality=kwargs.get("legality", AlwaysAllowLegality()),
    )


class _NonConcatenatingMergeTable:
    vocabulary_identity = object()

    def resolve(self, left: str, right: str) -> MergeRule | None:
        if (left, right) == (A, B):
            return MergeRule(result="wrong", result_id=7)
        return None


class FastMergeEngineProtocolTests(unittest.TestCase):
    """Protocol and interface conformance tests."""

    def test_implements_merge_engine_protocol(self):
        engine = FastMergeEngine()
        self.assertEqual(engine.name, "fast")
        self.assertTrue(isinstance(engine, MergeEngine))

    def test_get_merge_engine_resolver(self):
        ref = get_merge_engine("reference")
        self.assertIsInstance(ref, ReferenceMergeEngine)
        assert ref is not None
        self.assertEqual(ref.name, "reference")

        fast = get_merge_engine("fast")
        self.assertIsInstance(fast, FastMergeEngine)
        assert fast is not None
        self.assertEqual(fast.name, "fast")

        # Disable aliases return None
        for alias in ("default", "production", "none", "off"):
            self.assertIsNone(get_merge_engine(alias))

        # Passthrough
        self.assertIs(get_merge_engine(fast), fast)

        # Invalid name raises InvalidConfiguration
        with self.assertRaises(InvalidConfiguration):
            get_merge_engine("unknown_engine_xyz")

        with self.assertRaises(InvalidConfiguration):
            get_merge_engine(12345)  # type: ignore[arg-type]


class FastMergeEngineReviewVectorTests(unittest.TestCase):
    """Differential verification against ReferenceMergeEngine on all review vectors."""

    def setUp(self):
        self.ref_engine = ReferenceMergeEngine()
        self.fast_engine = FastMergeEngine()

    def _assert_engines_match(
        self,
        pieces: Sequence[str],
        results: Sequence[str],
        *,
        tape: Sequence[bool] | None = None,
        hard_cuts: Sequence[int] = (),
        legality=None,
    ) -> MergePlan:
        table = _table(results)
        constraints = _constraints(table, hard_cuts=hard_cuts, legality=legality)
        atoms = atoms_from_pieces(pieces)

        dec_ref = BooleanTapeDecisions(list(tape)) if tape is not None else None
        dec_fast = BooleanTapeDecisions(list(tape)) if tape is not None else None

        ref_plan = self.ref_engine.apply(atoms, table, constraints, dec_ref)
        fast_plan = self.fast_engine.apply(atoms, table, constraints, dec_fast)

        self.assertEqual(plan_to_pieces(fast_plan), plan_to_pieces(ref_plan))
        self.assertEqual(fast_plan.applied_merges, ref_plan.applied_merges)
        self.assertEqual(fast_plan.decisions_consumed, ref_plan.decisions_consumed)
        self.assertEqual(
            [(g.leaves, g.piece, g.merged_id) for g in fast_plan.groups],
            [(g.leaves, g.piece, g.merged_id) for g in ref_plan.groups],
        )
        return fast_plan

    def test_empty_and_singleton(self):
        empty_plan = self._assert_engines_match([], [A + B])
        self.assertEqual(empty_plan.groups, ())
        self.assertEqual(empty_plan.applied_merges, 0)
        self.assertEqual(empty_plan.decisions_consumed, 0)

        single_plan = self._assert_engines_match([A], [A + B])
        self.assertEqual(plan_to_pieces(single_plan), [A])
        self.assertEqual(single_plan.applied_merges, 0)
        self.assertEqual(single_plan.decisions_consumed, 0)
        self.assertIsNone(single_plan.groups[0].merged_id)

    def test_leftmost_overlap_ignores_ranks(self):
        table = MembershipMergeTable(
            eligible={A + B: 10, B + C: 99},
            vocabulary_identity=object(),
            ranks={(A, B): 100, (B, C): 1},
        )
        table = MembershipMergeTable(
            eligible=table.eligible,
            vocabulary_identity=frozenset(table.eligible.items()),
            ranks={(A, B): 100, (B, C): 1},
        )
        constraints = _constraints(table)
        atoms = atoms_from_pieces([A, B, C])

        ref_plan = self.ref_engine.apply(atoms, table, constraints, None)
        fast_plan = self.fast_engine.apply(atoms, table, constraints, None)

        self.assertEqual(plan_to_pieces(fast_plan), [A + B, C])
        self.assertEqual(plan_to_pieces(fast_plan), plan_to_pieces(ref_plan))
        self.assertEqual(fast_plan.applied_merges, 1)

    def test_pass_barrier(self):
        plan = self._assert_engines_match(
            [A, B, C, D],
            [A + B, B + C, A + B + C, C + D],
        )
        self.assertEqual(plan_to_pieces(plan), [A + B, C + D])
        self.assertEqual(plan.applied_merges, 2)
        self.assertEqual(plan.groups[0].leaves, (0, 2))
        self.assertEqual(plan.groups[1].leaves, (2, 4))

    def test_hierarchy_two_passes(self):
        plan = self._assert_engines_match(
            [A, B, C],
            [A + B, A + B + C],
        )
        self.assertEqual(plan_to_pieces(plan), [A + B + C])
        self.assertEqual(plan.applied_merges, 2)
        self.assertEqual(plan.groups[0].leaves, (0, 3))

    def test_different_decomposition(self):
        for pieces in ([A + B, C], [A, B + C]):
            plan = self._assert_engines_match(pieces, [A + B + C])
            self.assertEqual(plan_to_pieces(plan), [A + B + C])
            self.assertEqual(plan.groups[0].leaves, (0, 2))

    def test_persistent_dropout(self):
        plan = self._assert_engines_match(
            [A, B, C, D],
            [A + B, C + D],
            tape=[True, False],
        )
        self.assertEqual(plan_to_pieces(plan), [A, B, C + D])
        self.assertEqual(plan.decisions_consumed, 2)
        self.assertEqual(plan.applied_merges, 1)

    def test_changed_constituent_releases_block(self):
        plan = self._assert_engines_match(
            [A, B, C],
            [A + B, B + C, A + B + C],
            tape=[True, False, False],
        )
        self.assertEqual(plan_to_pieces(plan), [A + B + C])
        self.assertEqual(plan.decisions_consumed, 3)
        self.assertEqual(plan.applied_merges, 2)

    def test_leading_plus_internal_marker(self):
        left, right = S + "a", S + "b"
        merged = left + right
        plan = self._assert_engines_match([left, right], [merged])
        self.assertEqual(plan_to_pieces(plan), [merged])

    def test_custom_special_and_byte_membership(self):
        for left in ("<|s|>", "<0x61>"):
            merged = left + B
            plan = self._assert_engines_match([left, B], [merged])
            self.assertEqual(plan_to_pieces(plan), [merged])

    def test_empty_piece_consumes_leaf(self):
        ab = A + B
        plan = self._assert_engines_match([ab, ""], [ab])
        self.assertEqual(plan_to_pieces(plan), [ab])
        self.assertEqual(plan.groups[0].leaves, (0, 2))
        self.assertEqual(plan.applied_merges, 1)

    def test_shared_source_spans_preserved(self):
        leaves = [
            Token("<0xC3>", 1, (0, 1)),
            Token("<0xA9>", 2, (0, 1)),
        ]
        table = _table([])
        constraints = _constraints(table)
        atoms = atoms_from_pieces([t.text for t in leaves])
        ref_plan = self.ref_engine.apply(atoms, table, constraints, None)
        fast_plan = self.fast_engine.apply(atoms, table, constraints, None)
        self.assertEqual(plan_to_tokens(fast_plan, leaves), plan_to_tokens(ref_plan, leaves))

    def test_hard_cut_and_denied_legality(self):
        plan_cut = self._assert_engines_match([A, B], [A + B], hard_cuts=[1])
        self.assertEqual(plan_to_pieces(plan_cut), [A, B])
        self.assertEqual(plan_cut.decisions_consumed, 0)

        denied = PredicateLegality(lambda *_args: False)
        plan_deny = self._assert_engines_match([A, B], [A + B], legality=denied, tape=[True])
        self.assertEqual(plan_to_pieces(plan_deny), [A, B])
        self.assertEqual(plan_deny.decisions_consumed, 0)

    def test_stale_candidate_after_consumption(self):
        plan = self._assert_engines_match(
            [A, B, C, D],
            [A + B, B + C, C + D],
        )
        self.assertEqual(plan_to_pieces(plan), [A + B, C + D])
        self.assertEqual(plan.applied_merges, 2)


class FastMergeEngineErrorTests(unittest.TestCase):
    """Error and invalid configuration testing."""

    def setUp(self):
        self.engine = FastMergeEngine()
        self.table = _table([A + B])
        self.constraints = _constraints(self.table)

    def test_unsupported_profile_fails_loudly(self):
        bad = MergeConstraints(
            semantic_profile="RankFirstBpe",  # type: ignore[arg-type]
            vocabulary_identity=self.table.vocabulary_identity,
        )
        with self.assertRaises(UnsupportedSemantics):
            self.engine.apply(atoms_from_pieces([A, B]), self.table, bad, None)

    def test_identity_mismatch(self):
        bad = MergeConstraints(
            semantic_profile=SemanticProfile.SUPER_BPE_PASS_V1,
            vocabulary_identity=object(),
        )
        with self.assertRaises(InvalidConfiguration):
            self.engine.apply(atoms_from_pieces([A, B]), self.table, bad, None)

    def test_out_of_range_hard_cut(self):
        bad = _constraints(self.table, hard_cuts=[0])
        with self.assertRaises(InvalidConfiguration):
            self.engine.apply(atoms_from_pieces([A, B]), self.table, bad, None)

        bad2 = _constraints(self.table, hard_cuts=[2])
        with self.assertRaises(InvalidConfiguration):
            self.engine.apply(atoms_from_pieces([A, B]), self.table, bad2, None)

    def test_invalid_atom_piece_type(self):
        with self.assertRaises(InvalidConfiguration):
            self.engine.apply([Atom(piece=123)], self.table, self.constraints, None)  # type: ignore[arg-type]

    def test_decision_tape_exhaustion(self):
        table = _table([A + B, B + C, A + B + C])
        constraints = _constraints(table)
        tape = BooleanTapeDecisions([True, False])  # Needs 3 draws, only 2 provided
        with self.assertRaises(DecisionError):
            self.engine.apply(atoms_from_pieces([A, B, C]), table, constraints, tape)


class FastMergeEngineMalformedRuleTests(unittest.TestCase):
    def test_accepted_non_concatenating_result_matches_reference_error(self):
        table = _NonConcatenatingMergeTable()
        cases = (
            ("clean", None, None, 0),
            ("legality", AlwaysAllowLegality(), None, 0),
            ("dropout_accepts", None, [False], 1),
        )
        for case, legality, tape_values, expected_draws in cases:
            for engine in (ReferenceMergeEngine(), FastMergeEngine()):
                with self.subTest(case=case, engine=engine.name):
                    constraints = MergeConstraints(
                        semantic_profile=SemanticProfile.SUPER_BPE_PASS_V1,
                        vocabulary_identity=table.vocabulary_identity,
                        legality=legality,
                    )
                    decisions = BooleanTapeDecisions(tape_values) if tape_values is not None else None
                    with self.assertRaises(InvariantViolation) as raised:
                        engine.apply(atoms_from_pieces([A, B]), table, constraints, decisions)
                    self.assertEqual(
                        str(raised.exception),
                        "merge result 'wrong' is not concatenation of 'a' + '\u2581b'",
                    )
                    self.assertEqual(decisions.consumed if decisions is not None else 0, expected_draws)

    def test_rejected_non_concatenating_result_preserves_pieces_and_draws(self):
        table = _NonConcatenatingMergeTable()
        cases = (
            ("legality_denies", PredicateLegality(lambda *_args: False), [False], frozenset(), 0),
            ("hard_cut", None, [False], frozenset({1}), 0),
            ("dropout_rejects", None, [True], frozenset(), 1),
        )
        for case, legality, tape_values, hard_cuts, expected_draws in cases:
            for engine in (ReferenceMergeEngine(), FastMergeEngine()):
                with self.subTest(case=case, engine=engine.name):
                    constraints = MergeConstraints(
                        semantic_profile=SemanticProfile.SUPER_BPE_PASS_V1,
                        vocabulary_identity=table.vocabulary_identity,
                        hard_cuts=hard_cuts,
                        legality=legality,
                    )
                    decisions = BooleanTapeDecisions(tape_values)
                    plan = engine.apply(atoms_from_pieces([A, B]), table, constraints, decisions)
                    self.assertEqual(plan_to_pieces(plan), ["a", "\u2581b"])
                    self.assertEqual(plan.applied_merges, 0)
                    self.assertEqual(plan.decisions_consumed, expected_draws)
                    self.assertEqual(decisions.consumed, expected_draws)


class FastMergeEngineRandomizedParityTests(unittest.TestCase):
    """Reproducible randomized and adversarial differential tests."""

    def setUp(self):
        super().setUp()
        self._env_patcher = patch.dict(os.environ, {}, clear=False)
        self._env_patcher.start()
        os.environ.pop("UNIQTOKEN_MERGE_ENGINE", None)

    def tearDown(self):
        self._env_patcher.stop()
        super().tearDown()

    def test_seeded_random_tables_match_reference_and_production(self):
        rng = random.Random(2026)
        engine = FastMergeEngine()

        for trial in range(40):
            n = rng.randint(0, 10)
            pieces = [A if i % 2 == 0 else S + chr(ord("b") + (i // 2) % 6) for i in range(max(n, 0))]
            candidates: List[str] = []
            for i in range(len(pieces) - 1):
                candidates.append(pieces[i] + pieces[i + 1])
            for i in range(len(pieces) - 2):
                candidates.append(pieces[i] + pieces[i + 1] + pieces[i + 2])
            results = [c for c in candidates if S in c[1:] and c.strip(S) and rng.random() < 0.6]
            tok = _tokenizer(pieces or [A], results)

            # Test zero-dropout differential with FastMergeEngine
            report = differential_against_production(tok, pieces, 0.0, engine=engine)
            self.assertTrue(report["match"], msg=f"trial={trial} report={report}")

            if not results or len(pieces) < 2:
                continue

            # Test with dropout tape
            tape = [rng.random() < 0.4 for _ in range(64)]
            report_drop = differential_against_production(tok, pieces, 0.5, decision_tape=tape, engine=engine)
            self.assertTrue(report_drop["match"], msg=f"trial={trial} dropout report={report_drop}")

            # Test differential_reference_vs_fast helper directly
            table = cross_word_membership_table(tok)
            constraints = production_constraints(table)
            report_direct = differential_reference_vs_fast(
                pieces,
                table,
                constraints,
                decisions_factory=lambda: BooleanTapeDecisions(list(tape)),
            )
            self.assertTrue(report_direct["match"], msg=f"trial={trial} direct report={report_direct}")

    def test_long_sequence_parity(self):
        """Differential verification on a longer 500-token sequence with dense merges."""
        rng = random.Random(42)
        vocab_pool = [A, B, C, D]
        pieces = [rng.choice(vocab_pool) for _ in range(500)]
        results = [
            A + B,
            B + C,
            C + D,
            A + B + C,
            B + C + D,
            A + B + C + D,
            B + B,
            C + C,
            D + D,
        ]
        table = _table(results)
        constraints = _constraints(table)

        report = differential_reference_vs_fast(pieces, table, constraints, None)
        self.assertTrue(report["match"])
        applied = report["fast_applied"]
        self.assertIsInstance(applied, int)
        assert isinstance(applied, int)
        self.assertGreater(applied, 50)


class CustomTokenizerIntegrationTests(unittest.TestCase):
    """Verification of CustomTokenizer integration and engine selection."""

    def setUp(self):
        super().setUp()
        self._env_patcher = patch.dict(os.environ, {}, clear=False)
        self._env_patcher.start()
        os.environ.pop("UNIQTOKEN_MERGE_ENGINE", None)

    def tearDown(self):
        self._env_patcher.stop()
        super().tearDown()

    def test_default_engine_selection_unchanged(self):
        self.assertNotIn("UNIQTOKEN_MERGE_ENGINE", os.environ)
        tok = _tokenizer([A, B], [A + B])
        self.assertIsNone(tok.merge_engine)
        self.assertIsNone(tok._merge_engine_arg)
        merged_pieces = tok._apply_cross_word_merges([A, B])
        self.assertEqual(merged_pieces, [A + B])

    def test_explicit_fast_engine_selection(self):
        tok = _tokenizer([A, B], [A + B], merge_engine="fast")
        self.assertEqual(tok.merge_engine, "fast")
        pieces = tok._apply_cross_word_merges([A, B])
        self.assertEqual(pieces, [A + B])

    def test_explicit_reference_engine_selection(self):
        tok = _tokenizer([A, B], [A + B], merge_engine="reference")
        self.assertEqual(tok.merge_engine, "reference")
        pieces = tok._apply_cross_word_merges([A, B])
        self.assertEqual(pieces, [A + B])

    def test_parity_between_default_and_fast_engine(self):
        pieces = [A, B, C, D]
        results = [A + B, C + D, A + B + C + D]
        tok_default = _tokenizer(pieces, results)
        tok_fast = _tokenizer(pieces, results, merge_engine="fast")
        tok_ref = _tokenizer(pieces, results, merge_engine="reference")

        self.assertEqual(
            tok_default._apply_cross_word_merges(list(pieces)),
            tok_fast._apply_cross_word_merges(list(pieces)),
        )
        self.assertEqual(
            tok_fast._apply_cross_word_merges(list(pieces)),
            tok_ref._apply_cross_word_merges(list(pieces)),
        )

        leaves = [Token(p, tok_default.model.token_to_id[p], (i, i + 1)) for i, p in enumerate(pieces)]
        default_tokens = tok_default._apply_cross_word_merges_with_spans(list(leaves))
        fast_tokens = tok_fast._apply_cross_word_merges_with_spans(list(leaves))
        ref_tokens = tok_ref._apply_cross_word_merges_with_spans(list(leaves))

        self.assertEqual(
            [(t.text, t.id, t.raw_span) for t in default_tokens], [(t.text, t.id, t.raw_span) for t in fast_tokens]
        )
        self.assertEqual(
            [(t.text, t.id, t.raw_span) for t in fast_tokens], [(t.text, t.id, t.raw_span) for t in ref_tokens]
        )

    def test_set_merge_engine_toggle(self):
        tok = _tokenizer([A, B], [A + B])
        self.assertIsNone(tok.merge_engine)

        tok.set_merge_engine("fast")
        self.assertEqual(tok.merge_engine, "fast")
        self.assertEqual(tok._apply_cross_word_merges([A, B]), [A + B])

        tok.set_merge_engine("reference")
        self.assertEqual(tok.merge_engine, "reference")
        self.assertEqual(tok._apply_cross_word_merges([A, B]), [A + B])

        # Clean disable / fallback
        tok.set_merge_engine("default")
        self.assertIsNone(tok.merge_engine)

        tok.set_merge_engine(None)
        self.assertIsNone(tok.merge_engine)

        with self.assertRaises(InvalidConfiguration):
            tok.set_merge_engine("invalid_engine_name")
        self.assertIsNone(tok.merge_engine)
        self.assertIsNone(tok._merge_engine_arg)

    def test_set_merge_engine_failure_atomic(self):
        tok = _tokenizer([A, B], [A + B], merge_engine="fast")
        self.assertEqual(tok.merge_engine, "fast")
        self.assertEqual(tok._merge_engine_arg, "fast")
        with self.assertRaises(InvalidConfiguration):
            tok.set_merge_engine("invalid_engine_name")
        # Ensure previous valid engine and arg were not mutated
        self.assertEqual(tok.merge_engine, "fast")
        self.assertEqual(tok._merge_engine_arg, "fast")

    def test_environment_variable_override(self):
        with patch.dict(os.environ, {"UNIQTOKEN_MERGE_ENGINE": "fast"}):
            tok = _tokenizer([A, B], [A + B])
            self.assertEqual(tok.merge_engine, "fast")
            self.assertEqual(tok._apply_cross_word_merges([A, B]), [A + B])

        with patch.dict(os.environ, {"UNIQTOKEN_MERGE_ENGINE": "reference"}):
            tok = _tokenizer([A, B], [A + B])
            self.assertEqual(tok.merge_engine, "reference")
            self.assertEqual(tok._apply_cross_word_merges([A, B]), [A + B])

        with patch.dict(os.environ, {"UNIQTOKEN_MERGE_ENGINE": "default"}):
            tok = _tokenizer([A, B], [A + B])
            self.assertIsNone(tok.merge_engine)

        with patch.dict(os.environ, {"UNIQTOKEN_MERGE_ENGINE": "invalid_engine"}):
            with self.assertRaises(InvalidConfiguration):
                get_merge_engine(None)

    def test_serialization_round_trip(self):
        tok_fast = _tokenizer([A, B], [A + B], merge_engine="fast")
        with tempfile.TemporaryDirectory() as tmpdir:
            # JSON format
            tok_fast.save(tmpdir, save_binary=False)
            loaded = CustomTokenizer.load(tmpdir, prefer_binary=False)
            self.assertEqual(loaded.merge_engine, "fast")
            self.assertEqual(loaded._apply_cross_word_merges([A, B]), [A + B])

            # Binary format (.uniqtok)
            tok_fast.save(tmpdir, save_binary=True)
            loaded_bin = CustomTokenizer.load(tmpdir, prefer_binary=True)
            self.assertEqual(loaded_bin.merge_engine, "fast")
            self.assertEqual(loaded_bin._apply_cross_word_merges([A, B]), [A + B])

        tok_default = _tokenizer([A, B], [A + B])
        with tempfile.TemporaryDirectory() as tmpdir:
            tok_default.save(tmpdir, save_binary=False)
            loaded_default = CustomTokenizer.load(tmpdir, prefer_binary=False)
            self.assertIsNone(loaded_default.merge_engine)

            tok_default.save(tmpdir, save_binary=True)
            loaded_default_bin = CustomTokenizer.load(tmpdir, prefer_binary=True)
            self.assertIsNone(loaded_default_bin.merge_engine)

    def test_custom_engine_instance_warning_on_save(self):
        import warnings

        fast_inst = FastMergeEngine()
        tok = _tokenizer([A, B], [A + B], merge_engine=fast_inst)
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                tok.save(tmpdir, save_binary=False)
                self.assertTrue(any(issubclass(w.category, UserWarning) for w in recorded))

    def test_explicit_load_merge_engine_override(self):
        tok_fast = _tokenizer([A, B], [A + B], merge_engine="fast")
        with tempfile.TemporaryDirectory() as tmpdir:
            tok_fast.save(tmpdir, save_binary=True)
            # Override with reference engine
            loaded_ref = CustomTokenizer.load(tmpdir, prefer_binary=True, merge_engine="reference")
            self.assertEqual(loaded_ref.merge_engine, "reference")
            # Override with default inlined
            loaded_def = CustomTokenizer.load(tmpdir, prefer_binary=True, merge_engine="default")
            self.assertIsNone(loaded_def.merge_engine)


if __name__ == "__main__":
    unittest.main()
