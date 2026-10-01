"""Tests for ReferenceMergeEngine correctness oracle (#108).

Covers the SuperBpePassV1 semantic review vectors from the MergeEngine
design, differential parity against production helpers, hard cuts / legality,
stale-candidate / tie behaviour, and reproducible randomised adversarial cases.
"""

from __future__ import annotations

import random
import unittest
from typing import Dict, List, Sequence
from unittest.mock import patch

from uniqtoken.merge_engine import (
    AlwaysAllowLegality,
    BooleanTapeDecisions,
    DecisionError,
    InvalidConfiguration,
    MembershipMergeTable,
    MergeConstraints,
    MergePlan,
    PredicateLegality,
    ReferenceMergeEngine,
    SemanticProfile,
    UnsupportedSemantics,
    apply_reference_to_pieces,
    apply_reference_to_tokens,
    atoms_from_pieces,
    cross_word_membership_table,
    differential_against_production,
    document_semantic_invariants,
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


def _tokenizer(pieces: Sequence[str], results: Sequence[str]) -> CustomTokenizer:
    model = _model_from_pieces(pieces, results)
    return CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)


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


class SemanticInvariantDocTests(unittest.TestCase):
    def test_invariants_document_is_nonempty(self):
        text = document_semantic_invariants()
        self.assertIn("SuperBpePassV1", text)
        self.assertIn("left-to-right", text)
        self.assertIn("UnsupportedSemantics", text)


class ReviewVectorTests(unittest.TestCase):
    """Focused unit tests for every documented review vector."""

    def setUp(self):
        self.engine = ReferenceMergeEngine()

    def _run(
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
        decisions = BooleanTapeDecisions(list(tape)) if tape is not None else None
        return self.engine.apply(atoms_from_pieces(pieces), table, constraints, decisions)

    def test_empty_and_singleton(self):
        empty = self._run([], [A + B])
        self.assertEqual(empty.groups, ())
        self.assertEqual(empty.applied_merges, 0)
        self.assertEqual(empty.decisions_consumed, 0)

        single = self._run([A], [A + B])
        self.assertEqual(plan_to_pieces(single), [A])
        self.assertEqual(single.applied_merges, 0)
        self.assertEqual(single.decisions_consumed, 0)
        self.assertEqual(single.groups[0].leaves, (0, 1))
        self.assertIsNone(single.groups[0].merged_id)

    def test_leftmost_overlap_ignores_ranks(self):
        table = MembershipMergeTable(
            eligible={A + B: 10, B + C: 99},
            vocabulary_identity=object(),
            ranks={(A, B): 100, (B, C): 1},
        )
        # Re-bind identity for constraints.
        table = MembershipMergeTable(
            eligible=table.eligible,
            vocabulary_identity=frozenset(table.eligible.items()),
            ranks={(A, B): 100, (B, C): 1},
        )
        plan = self.engine.apply(
            atoms_from_pieces([A, B, C]),
            table,
            _constraints(table),
            None,
        )
        self.assertEqual(plan_to_pieces(plan), [A + B, C])
        self.assertEqual(plan.applied_merges, 1)

    def test_pass_barrier(self):
        # Eager AB -> ABC would steal C from CD; pass barrier yields [AB, CD].
        plan = self._run([A, B, C, D], [A + B, B + C, A + B + C, C + D])
        self.assertEqual(plan_to_pieces(plan), [A + B, C + D])
        self.assertEqual(plan.applied_merges, 2)
        self.assertEqual(plan.groups[0].leaves, (0, 2))
        self.assertEqual(plan.groups[1].leaves, (2, 4))

    def test_hierarchy_two_passes(self):
        plan = self._run([A, B, C], [A + B, A + B + C])
        self.assertEqual(plan_to_pieces(plan), [A + B + C])
        self.assertEqual(plan.applied_merges, 2)
        self.assertEqual(plan.groups[0].leaves, (0, 3))

    def test_different_decomposition(self):
        for pieces in ([A + B, C], [A, B + C]):
            plan = self._run(pieces, [A + B + C])
            self.assertEqual(plan_to_pieces(plan), [A + B + C])
            self.assertEqual(plan.groups[0].leaves, (0, 2))

    def test_persistent_dropout(self):
        # Drop AB, accept CD; AB stays blocked across the unrelated CD merge.
        plan = self._run(
            [A, B, C, D],
            [A + B, C + D],
            tape=[True, False],
        )
        self.assertEqual(plan_to_pieces(plan), [A, B, C + D])
        self.assertEqual(plan.decisions_consumed, 2)
        self.assertEqual(plan.applied_merges, 1)

    def test_changed_constituent_releases_block(self):
        plan = self._run(
            [A, B, C],
            [A + B, B + C, A + B + C],
            tape=[True, False, False],
        )
        self.assertEqual(plan_to_pieces(plan), [A + B + C])
        self.assertEqual(plan.decisions_consumed, 3)
        self.assertEqual(plan.applied_merges, 2)

    def test_marker_membership_leading_only_excluded_by_table(self):
        # Leading-only marker piece 'S'+'b' built via tokenizer table is excluded.
        tok = _tokenizer([S, "b"], [S + "b"])
        table = cross_word_membership_table(tok)
        self.assertNotIn(S + "b", table.eligible)
        plan = self.engine.apply(
            atoms_from_pieces([S, "b"]),
            table,
            production_constraints(table),
            None,
        )
        self.assertEqual(plan_to_pieces(plan), [S, "b"])

    def test_leading_plus_internal_marker_merges(self):
        left, right = S + "a", S + "b"
        merged = left + right
        plan = self._run([left, right], [merged])
        self.assertEqual(plan_to_pieces(plan), [merged])

    def test_custom_special_and_byte_membership(self):
        special = "<|s|>"
        byte_tok = "<0x61>"
        for left in (special, byte_tok):
            merged = left + B
            plan = self._run([left, B], [merged])
            self.assertEqual(plan_to_pieces(plan), [merged])

    def test_empty_piece_consumes_leaf(self):
        ab = A + B
        plan = self._run([ab, ""], [ab])
        self.assertEqual(plan_to_pieces(plan), [ab])
        self.assertEqual(plan.groups[0].leaves, (0, 2))
        self.assertEqual(plan.applied_merges, 1)

    def test_shared_source_spans_preserved(self):
        # Two byte-fallback leaves sharing one character span; no merge rule.
        leaves = [
            Token("<0xC3>", 1, (0, 1)),
            Token("<0xA9>", 2, (0, 1)),
        ]
        table = _table([])
        plan = self.engine.apply(
            atoms_from_pieces([t.text for t in leaves]),
            table,
            _constraints(table),
            None,
        )
        projected = plan_to_tokens(plan, leaves)
        self.assertEqual([(t.text, t.id, t.raw_span) for t in projected], [(t.text, t.id, t.raw_span) for t in leaves])

    def test_hard_cut_and_denied_legality(self):
        plan_cut = self._run([A, B], [A + B], hard_cuts=[1])
        self.assertEqual(plan_to_pieces(plan_cut), [A, B])
        self.assertEqual(plan_cut.decisions_consumed, 0)

        denied = PredicateLegality(lambda *_args: False)
        plan_deny = self._run([A, B], [A + B], legality=denied, tape=[True])
        self.assertEqual(plan_to_pieces(plan_deny), [A, B])
        # Legality rejects before any dropout draw.
        self.assertEqual(plan_deny.decisions_consumed, 0)

    def test_stale_candidate_after_consumption(self):
        # After AB consumes B, a conceptual BC candidate is gone; only CD remains.
        plan = self._run([A, B, C, D], [A + B, B + C, C + D])
        self.assertEqual(plan_to_pieces(plan), [A + B, C + D])
        self.assertEqual(plan.applied_merges, 2)


class ErrorAndConfigTests(unittest.TestCase):
    def setUp(self):
        self.engine = ReferenceMergeEngine()
        self.table = _table([A + B])
        self.constraints = _constraints(self.table)

    def test_unsupported_profile(self):
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

    def test_decision_tape_exhaustion(self):
        # Drop AB, accept BC (releases AB), then A+BC needs a third draw.
        table = _table([A + B, B + C, A + B + C])
        constraints = _constraints(table)
        tape = BooleanTapeDecisions([True, False])  # only two draws
        with self.assertRaises(DecisionError):
            self.engine.apply(
                atoms_from_pieces([A, B, C]),
                table,
                constraints,
                tape,
            )


class DifferentialParityTests(unittest.TestCase):
    """Differential parity against both unmodified production helpers."""

    def test_design_probe_cases(self):
        cases = [
            ([A, B, C], [A + B, B + C], [A + B, C], 0.0, []),
            ([A, B, C, D], [A + B, B + C, A + B + C, C + D], [A + B, C + D], 0.0, []),
            ([A, B, C], [A + B, A + B + C], [A + B + C], 0.0, []),
            ([A, B, C, D], [A + B, C + D], [A, B, C + D], 0.5, [True, False]),
            ([A, B, C], [A + B, B + C, A + B + C], [A + B + C], 0.5, [True, False, False]),
            ([A + B, ""], [A + B], [A + B], 0.0, []),
        ]
        for pieces, results, expected, probability, tape in cases:
            with self.subTest(pieces=pieces, expected=expected, tape=tape):
                tok = _tokenizer(pieces, results)
                if probability == 0.0:
                    report = differential_against_production(tok, pieces, 0.0)
                else:
                    report = differential_against_production(tok, pieces, probability, decision_tape=tape)
                self.assertTrue(report["match"], msg=report)
                self.assertEqual(report["reference_pieces"], expected)
                self.assertEqual(report["production_pieces"], expected)
                self.assertEqual(report["decisions_consumed"], len(tape))

    def test_parity_empty_cross_word_set(self):
        tok = _tokenizer([A, B], [])  # B has leading metaspace only; not cross-word
        # Force a vocab without internal metaspace merges.
        report = differential_against_production(tok, [A, "x", B], 0.0)
        self.assertTrue(report["match"])
        self.assertEqual(report["reference_pieces"], [A, "x", B])

    def test_span_projection_matches_production(self):
        pieces = [A, B, C]
        results = [A + B, A + B + C]
        tok = _tokenizer(pieces, results)
        leaves = [Token(p, tok.model.token_to_id[p], (i * 2, i * 2 + 1)) for i, p in enumerate(pieces)]
        engine = ReferenceMergeEngine()
        table = cross_word_membership_table(tok)
        ref_tokens, plan = apply_reference_to_tokens(engine, leaves, table, production_constraints(table), None)
        prod = tok._apply_cross_word_merges_with_spans(list(leaves), 0.0)
        self.assertEqual([t.text for t in ref_tokens], [t.text for t in prod])
        self.assertEqual([t.id for t in ref_tokens], [t.id for t in prod])
        self.assertEqual([t.raw_span for t in ref_tokens], [t.raw_span for t in prod])
        self.assertEqual(plan.applied_merges, len(pieces) - len(ref_tokens))


class RandomizedAdversarialTests(unittest.TestCase):
    """Reproducible randomised and adversarial sequences."""

    def test_seeded_random_tables_match_production(self):
        rng = random.Random(108)
        engine = ReferenceMergeEngine()
        for trial in range(40):
            n = rng.randint(0, 8)
            # Build pieces that can form cross-word concatenations.
            pieces = [A if i % 2 == 0 else S + chr(ord("b") + (i // 2) % 6) for i in range(max(n, 0))]
            # Sample a subset of adjacent concatenations (and some longer).
            candidates: List[str] = []
            for i in range(len(pieces) - 1):
                candidates.append(pieces[i] + pieces[i + 1])
            for i in range(len(pieces) - 2):
                candidates.append(pieces[i] + pieces[i + 1] + pieces[i + 2])
            results = [c for c in candidates if S in c[1:] and c.strip(S) and rng.random() < 0.55]
            tok = _tokenizer(pieces or [A], results)
            # Zero-dropout differential.
            report = differential_against_production(tok, pieces, 0.0)
            self.assertTrue(report["match"], msg=f"trial={trial} report={report}")

            if not results or len(pieces) < 2:
                continue
            # Positive dropout with a shared Boolean tape long enough for worst case.
            tape = [rng.random() < 0.4 for _ in range(64)]
            report = differential_against_production(tok, pieces, 0.5, decision_tape=tape)
            self.assertTrue(report["match"], msg=f"trial={trial} dropout report={report}")
            # Reference alone must be deterministic for the same tape.
            table = cross_word_membership_table(tok)
            constraints = production_constraints(table)
            p1, plan1 = apply_reference_to_pieces(engine, pieces, table, constraints, BooleanTapeDecisions(list(tape)))
            p2, plan2 = apply_reference_to_pieces(engine, pieces, table, constraints, BooleanTapeDecisions(list(tape)))
            self.assertEqual(p1, p2)
            self.assertEqual(plan1.decisions_consumed, plan2.decisions_consumed)

    def test_adversarial_repeated_and_pathological(self):
        engine = ReferenceMergeEngine()
        # Repeated identical metaspace-crossing candidates.
        pieces = [A, B, A, B, A, B]
        results = [A + B, B + A, A + B + A, A + B + A + B]
        tok = _tokenizer(pieces, results)
        report = differential_against_production(tok, pieces, 0.0)
        self.assertTrue(report["match"], msg=report)

        # Pathological: many empty pieces interleaved.
        pieces2 = [A + B, "", A + B, "", ""]
        results2 = [A + B]
        tok2 = _tokenizer(pieces2, results2)
        report2 = differential_against_production(tok2, pieces2, 0.0)
        self.assertTrue(report2["match"], msg=report2)

        # Hard cuts between every leaf: zero merges, zero draws.
        table = _table([A + B, B + C, C + D, A + B + C + D])
        cuts = frozenset(range(1, 4))
        plan = engine.apply(
            atoms_from_pieces([A, B, C, D]),
            table,
            _constraints(table, hard_cuts=cuts),
            BooleanTapeDecisions([False] * 10),
        )
        self.assertEqual(plan_to_pieces(plan), [A, B, C, D])
        self.assertEqual(plan.decisions_consumed, 0)


class ProductionPythonRngParityTests(unittest.TestCase):
    def test_python_random_decisions_draw_count(self):
        pieces = [A, B, C, D]
        results = [A + B, C + D]
        tok = _tokenizer(pieces, results)
        draws = [0.0, 0.99]
        with patch("random.random", side_effect=list(draws)) as rng:
            prod = tok._apply_cross_word_merges(list(pieces), dropout_prob=0.5)
        self.assertEqual(rng.call_count, 2)

        engine = ReferenceMergeEngine()
        table = cross_word_membership_table(tok)
        from uniqtoken.merge_engine import PythonRandomDecisions

        with patch("random.random", side_effect=list(draws)) as rng2:
            decisions = PythonRandomDecisions(0.5)
            plan = engine.apply(
                atoms_from_pieces(pieces),
                table,
                production_constraints(table),
                decisions,
            )
        self.assertEqual(rng2.call_count, 2)
        self.assertEqual(plan.decisions_consumed, 2)
        self.assertEqual(plan_to_pieces(plan), prod)
        self.assertEqual(decisions.draws, 2)


class PartitionInvariantTests(unittest.TestCase):
    def test_applied_merges_equals_leaf_delta(self):
        engine = ReferenceMergeEngine()
        pieces = [A, B, C, D]
        table = _table([A + B, C + D, A + B + C + D])
        plan = engine.apply(atoms_from_pieces(pieces), table, _constraints(table), None)
        self.assertEqual(plan.applied_merges, len(pieces) - len(plan.groups))
        cursor = 0
        for group in plan.groups:
            self.assertEqual(group.start, cursor)
            self.assertGreater(group.end, group.start)
            self.assertEqual(group.piece, "".join(pieces[group.start : group.end]))
            cursor = group.end
        self.assertEqual(cursor, len(pieces))


if __name__ == "__main__":
    unittest.main()
