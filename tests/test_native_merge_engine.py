"""Actual native execution against the SuperBpePassV1 oracle; no fallback skips."""

from __future__ import annotations

import importlib
import importlib.util
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from uniqtoken.merge_engine import (
    AlwaysAllowLegality,
    BooleanTapeDecisions,
    DecisionError,
    InvalidConfiguration,
    MembershipMergeTable,
    MergeConstraints,
    PredicateLegality,
    PythonRandomDecisions,
    ReferenceMergeEngine,
    SemanticProfile,
    UnsupportedSemantics,
    atoms_from_pieces,
    plan_to_tokens,
)
from uniqtoken.tokenizer import Token


class NativeMergeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("uniqtoken.native_merge_engine"), "native prototype is not implemented"
        )
        self.module = importlib.import_module("uniqtoken.native_merge_engine")
        self.engine = self.module.NativeFastMergeEngine()

    def compare(self, pieces, eligible, *, cuts=(), tape=None, cursor=0, ranks=None):
        table = MembershipMergeTable(eligible, object(), ranks or {})
        constraints = MergeConstraints(SemanticProfile.SUPER_BPE_PASS_V1, table.vocabulary_identity, frozenset(cuts))
        ref_dec = BooleanTapeDecisions(tape, cursor) if tape is not None else None
        native_dec = BooleanTapeDecisions(tape, cursor) if tape is not None else None
        atoms = atoms_from_pieces(pieces)
        expected = ReferenceMergeEngine().apply(atoms, table, constraints, ref_dec)
        actual = self.engine.apply(atoms, table, constraints, native_dec)
        self.assertEqual(actual, expected)
        if native_dec is not None:
            self.assertEqual(native_dec.consumed, ref_dec.consumed)
        return actual

    def test_private_native_entry_is_not_a_package_export(self):
        package = importlib.import_module("uniqtoken_core")
        self.assertFalse(hasattr(package, "_experimental_merge_plan"))
        self.assertNotIn("_experimental_merge_plan", package.__all__)

    def test_native_entry_and_no_reference_fallback(self):
        native = importlib.import_module("uniqtoken_core.uniqtoken_core")
        self.assertTrue(callable(getattr(native, "_experimental_merge_plan", None)))
        table = MembershipMergeTable({"ab": 42}, object())
        constraints = MergeConstraints(SemanticProfile.SUPER_BPE_PASS_V1, table.vocabulary_identity)
        with patch.object(ReferenceMergeEngine, "apply", side_effect=AssertionError("reference fallback")):
            plan = self.engine.apply(atoms_from_pieces(["a", "b"]), table, constraints)
        self.assertEqual([(g.leaves, g.piece, g.merged_id) for g in plan.groups], [((0, 2), "ab", 42)])

    def test_pass_barrier_and_leftmost_overlap_ignore_ranks(self):
        plan = self.compare(
            ["a", "b", "c", "d"], {"ab": 100, "abc": 300, "cd": 200}, ranks={("c", "d"): 0, ("a", "b"): 99}
        )
        self.assertEqual([g.piece for g in plan.groups], ["ab", "cd"])
        self.assertEqual(plan.applied_merges, 2)
        plan = self.compare(["a", "b", "c"], {"ab": 9, "bc": 1})
        self.assertEqual([g.piece for g in plan.groups], ["ab", "c"])

    def test_dropout_changed_neighbor_unblocks_but_position_does_not(self):
        self.compare(["a", "b", "c", "d"], {"ab": 1, "bc": 2, "abc": 3}, tape=[True, False, False])
        plan = self.compare(["a", "b", "c", "d", "e"], {"ab": 1, "de": 2}, tape=[True, False])
        self.assertEqual(plan.decisions_consumed, 2)
        self.assertEqual([g.piece for g in plan.groups], ["a", "b", "c", "de"])

    def test_empty_repeated_unicode_special_and_fallback_pieces(self):
        for pieces in ([], [""], ["", "", "a", ""], ["a"] * 64, ["é", "▁中", "👩‍💻", "<s>", "<0xC3>", "<0xA9>"]):
            eligible = {
                "".join(pieces[i:j]): 100 + i * 100 + j
                for i in range(len(pieces))
                for j in range(i + 2, len(pieces) + 1)
            }
            with self.subTest(pieces=pieces[:8]):
                self.compare(pieces, eligible)

    def test_hard_cuts_and_sparse_u64_ids(self):
        plan = self.compare(["a", "b", "c", "d"], {"ab": 2**64 - 1, "cd": 0, "abcd": 999}, cuts=[2])
        self.assertEqual([g.merged_id for g in plan.groups], [2**64 - 1, 0])

    def test_projection_preserves_overlapping_source_spans(self):
        leaves = [Token("<0xC3>", 1, (0, 1)), Token("<0xA9>", 2, (0, 1)), Token("▁中", 3, (1, 4))]
        original = [(t.text, t.id, t.raw_span) for t in leaves]
        plan = self.compare([t.text for t in leaves], {"<0xC3><0xA9>": 77})
        projected = plan_to_tokens(plan, leaves)
        self.assertEqual(
            [(t.text, t.id, t.raw_span) for t in projected], [("<0xC3><0xA9>", 77, (0, 1)), ("▁中", 3, (1, 4))]
        )
        self.assertIs(projected[1], leaves[2])
        self.assertEqual([(t.text, t.id, t.raw_span) for t in leaves], original)

    def test_partial_cursor_and_exhaustion_match_oracle(self):
        self.compare(["a", "b", "c"], {"ab": 1, "bc": 2}, tape=[False, True, False], cursor=1)
        for tape, cursor in (([], 0), ([True], 1), ([False], 0), ([False, False], 1)):
            table = MembershipMergeTable({"ab": 1, "abc": 2}, object())
            constraints = MergeConstraints(SemanticProfile.SUPER_BPE_PASS_V1, table.vocabulary_identity)
            outcomes = []
            for engine in (ReferenceMergeEngine(), self.engine):
                decisions = BooleanTapeDecisions(tape, cursor)
                with self.assertRaises(DecisionError) as error:
                    engine.apply(atoms_from_pieces(["a", "b", "c"]), table, constraints, decisions)
                outcomes.append((str(error.exception), decisions.consumed))
            self.assertEqual(outcomes[0], outcomes[1])

    def test_unsupported_capabilities_fail_before_decisions(self):
        class CustomTable(MembershipMergeTable):
            pass

        class CustomAllow(AlwaysAllowLegality):
            pass

        class CustomTape(BooleanTapeDecisions):
            pass

        table = MembershipMergeTable({"ab": 1}, object())
        allow_override = AlwaysAllowLegality()
        allow_override.allows = lambda *args: False
        tape_override = BooleanTapeDecisions([False])
        tape_override.drop_next = lambda: True
        for candidate, legality, decisions in (
            (CustomTable({"ab": 1}, object()), None, BooleanTapeDecisions([False])),
            (table, CustomAllow(), BooleanTapeDecisions([False])),
            (table, allow_override, BooleanTapeDecisions([False])),
            (table, PredicateLegality(lambda *args: False), BooleanTapeDecisions([False])),
            (table, None, CustomTape([False])),
            (table, None, tape_override),
            (table, None, PythonRandomDecisions(0.5)),
        ):
            constraints = MergeConstraints(
                SemanticProfile.SUPER_BPE_PASS_V1, candidate.vocabulary_identity, legality=legality
            )
            with self.subTest(table=type(candidate), legality=type(legality), decisions=type(decisions)):
                with self.assertRaises(UnsupportedSemantics):
                    self.engine.apply(atoms_from_pieces(["a", "b"]), candidate, constraints, decisions)
                self.assertEqual(getattr(decisions, "consumed", getattr(decisions, "draws", 0)), 0)

    def test_invalid_configuration_and_unsupported_width_or_text(self):
        for pieces, eligible, cuts, profile, identity_ok, cursor, error in (
            (["a", "b"], {"ab": -1}, (), SemanticProfile.SUPER_BPE_PASS_V1, True, 0, InvalidConfiguration),
            (["a", "b"], {"ab": True}, (), SemanticProfile.SUPER_BPE_PASS_V1, True, 0, InvalidConfiguration),
            (["a", "b"], {"ab": 2**64}, (), SemanticProfile.SUPER_BPE_PASS_V1, True, 0, UnsupportedSemantics),
            (["\ud800", "b"], {}, (), SemanticProfile.SUPER_BPE_PASS_V1, True, 0, UnsupportedSemantics),
            (["a", "b"], {"ab": 1}, (0,), SemanticProfile.SUPER_BPE_PASS_V1, True, 0, InvalidConfiguration),
            (["a", "b"], {"ab": 1}, (True,), SemanticProfile.SUPER_BPE_PASS_V1, True, 0, InvalidConfiguration),
            ([], {}, (), "RankFirst", True, 0, UnsupportedSemantics),
            (["a"], {}, (), SemanticProfile.SUPER_BPE_PASS_V1, False, 0, InvalidConfiguration),
            (["a"], {}, (), SemanticProfile.SUPER_BPE_PASS_V1, True, -1, InvalidConfiguration),
        ):
            table = MembershipMergeTable(eligible, object())
            constraints = MergeConstraints(
                profile, table.vocabulary_identity if identity_ok else object(), frozenset(cuts)
            )
            decisions = BooleanTapeDecisions([False], cursor)
            with self.subTest(pieces=ascii(pieces), eligible=eligible, cuts=cuts, profile=profile):
                with self.assertRaises(error):
                    self.engine.apply(atoms_from_pieces(pieces), table, constraints, decisions)
                self.assertEqual(decisions.consumed, cursor)

    def test_missing_native_symbol_is_not_a_fallback(self):
        table = MembershipMergeTable({"ab": 1}, object())
        constraints = MergeConstraints(SemanticProfile.SUPER_BPE_PASS_V1, table.vocabulary_identity)
        with patch.object(self.module.importlib, "import_module", return_value=SimpleNamespace()):
            with self.assertRaises(UnsupportedSemantics):
                self.engine.apply(atoms_from_pieces(["a", "b"]), table, constraints)

    def test_custom_container_semantics_are_not_replaced_by_snapshots(self):
        class CustomMap(dict):
            def get(self, key, default=None):
                return default

        class CustomTape(list):
            pass

        class CustomCuts(frozenset):
            pass

        for eligible, cuts, tape in (
            (CustomMap(ab=1), frozenset(), [False]),
            ({"ab": 1}, CustomCuts(), [False]),
            ({"ab": 1}, frozenset(), CustomTape([False])),
        ):
            table = MembershipMergeTable(eligible, object())
            constraints = MergeConstraints(SemanticProfile.SUPER_BPE_PASS_V1, table.vocabulary_identity, cuts)
            decisions = BooleanTapeDecisions(tape)
            with self.subTest(container=(type(eligible), type(cuts), type(tape))):
                with self.assertRaises(UnsupportedSemantics):
                    self.engine.apply(atoms_from_pieces(["a", "b"]), table, constraints, decisions)
                self.assertEqual(decisions.consumed, 0)

    def test_overridden_class_method_is_unsupported(self):
        table = MembershipMergeTable({"ab": 1}, object())
        constraints = MergeConstraints(SemanticProfile.SUPER_BPE_PASS_V1, table.vocabulary_identity)
        with patch.object(MembershipMergeTable, "resolve", return_value=None):
            with self.assertRaises(UnsupportedSemantics):
                self.engine.apply(atoms_from_pieces(["a", "b"]), table, constraints)

    def test_seeded_random_differential(self):
        rng = random.Random(1092026)
        for trial in range(400):
            pieces = [rng.choice(["", "a", "▁b", "é", "中", "<0xFF>"]) for _ in range(rng.randrange(35))]
            eligible = {
                "".join(pieces[i:j]): 1000 + i * 100 + j
                for i in range(len(pieces))
                for j in range(i + 2, min(i + 6, len(pieces)) + 1)
                if rng.random() < 0.3
            }
            cuts = [i for i in range(1, len(pieces)) if rng.random() < 0.1]
            tape = [rng.random() < 0.3 for _ in range(4 * len(pieces))] if trial % 2 else None
            with self.subTest(trial=trial):
                self.compare(pieces, eligible, cuts=cuts, tape=tape)
