"""Independent candidate accounting for the offline component experiment."""

import copy
import math
from pathlib import Path
import unittest

from benchmarks import merge_objective_ablation as m
from benchmarks import run_research_experiments as h
from tests.test_byte_fallback_analysis import byte_model
from uniqtoken.cem_merger import SuperBPE
from uniqtoken.unigram_trainer import UnigramModel


def inputs():
    docs = [["a\u2581b", h.SPECIALS[3]]] * 4 + [["b\u2581a", h.SPECIALS[3]]] * 3
    chunks = [c for doc in docs for c in doc]
    model = byte_model()
    return model, chunks, docs, m.candidate_pool(model, chunks, docs)


class ObjectiveAblationTests(unittest.TestCase):
    def test_retained_matrix_is_receipted_and_uses_one_candidate_pool(self):
        root = Path(m.__file__).parent / "objective_ablation" / "issue87"
        receipt = h.read_json(root / "manifest.json")
        self.assertEqual(receipt["status"], "complete")
        for path, digest in receipt["artifacts"].items():
            self.assertEqual(h.file_hash(root / path), digest, path)
        result = h.read_json(root / "results.json")
        self.assertEqual(h.digest(result["pool"]), result["pool_sha256"])
        self.assertEqual(set(result["conditions"]), {"current_superbpe", *m.MATRIX})
        self.assertEqual(result["assignments"]["source"]["test_access"], "forbidden_not_opened")
        self.assertFalse(result["identity"]["working_tree_dirty"])
        self.assertEqual(result["constant_components"], ["fallback_cost"])
        for name, condition in result["conditions"].items():
            self.assertEqual(condition["actual_vocab_size"], 8192)
            self.assertEqual(len(condition["selected"]), 64)
            self.assertEqual(h.artifact_hashes(root / name), condition["model_hashes"])
            if name != "current_superbpe":
                self.assertEqual(m.select_candidates(result["pool"], name, 64), condition["selected"])
        self.assertEqual(
            h.file_hash(root / "current_superbpe" / "tokenizer.json"),
            "bcabbdeddb1bb4054234cd9b034ae62c8d75875c2a169aca9ff3cd45cb47960f",
        )

    def test_initial_ce_selection_matches_current_superbpe(self):
        model, chunks, _, pool = inputs()
        selected = m.select_candidates(pool, "pool_ce", 1)
        optimizer = SuperBPE(max_merges=1)
        optimizer.optimize(model, chunks)
        self.assertEqual(selected[0]["merged"], optimizer.merges[0][2])
        self.assertAlmostEqual(selected[0]["cross_entropy"], optimizer.merges[0][3])
        self.assertEqual(selected[0]["frequency"], optimizer.merges[0][4])

    def test_frequency_and_compression_are_distinct_units(self):
        model = byte_model()
        docs = [["a\u2581ba\u2581b", h.SPECIALS[3]]] * 3
        pool = m.candidate_pool(model, [c for doc in docs for c in doc], docs)
        row = next(r for r in pool if r["left"] == "a" and r["right"] == "\u2581")
        self.assertEqual(row["frequency"], 6)
        self.assertEqual(row["compression_gain"], 6)
        self.assertEqual(row["document_frequency"], 3)
        self.assertEqual(row["boundary_cost"], 6)
        self.assertEqual(row["fragmentation_penalty"], 0)

    def test_nonoverlapping_gain_for_repeated_tokens(self):
        model = byte_model()
        ids = {**model.token_to_id, "a\u2581": len(model.vocab)}
        model = UnigramModel({**model.vocab, "a\u2581": -0.1}, ids, {i: t for t, i in ids.items()}, list(h.SPECIALS))
        docs = [["a\u2581" * 4]]
        pool = m.candidate_pool(model, docs[0], docs)
        row = next(r for r in pool if r["left"] == r["right"] == "a\u2581")
        self.assertEqual(row["frequency"], 3)
        self.assertEqual(row["compression_gain"], 2)

    def test_no_fallback_or_control_candidates(self):
        model, chunks, docs, _ = inputs()
        docs += [["\u093e\u093e", h.SPECIALS[3]]] * 3
        pool = m.candidate_pool(model, [c for doc in docs for c in doc], docs)
        self.assertTrue(pool)
        self.assertTrue(all(row["fallback_cost"] == 0 for row in pool))
        self.assertTrue(
            all(
                not any(m.Bytes.is_byte_token(row[k]) or row[k] in model.special_tokens for k in ("left", "right"))
                for row in pool
            )
        )

    def test_all_components_share_pool_and_preserve_existing_ids(self):
        model, _, _, pool = inputs()
        original = copy.deepcopy(pool)
        for condition in m.MATRIX:
            selected = m.select_candidates(pool, condition, 2)
            updated = m.admit(model, selected)
            self.assertEqual(len(updated.vocab), len(model.vocab) + 2)
            self.assertTrue(all(updated.token_to_id[t] == i for t, i in model.token_to_id.items()))
            self.assertAlmostEqual(math.fsum(math.exp(p) for p in updated.vocab.values()), 1)
            self.assertEqual(pool, original)

    def test_order_determinism_and_no_duplicate_slots(self):
        _, _, _, pool = inputs()
        for condition in m.MATRIX:
            self.assertEqual(
                m.select_candidates(pool, condition, 2), m.select_candidates(list(reversed(pool)), condition, 2)
            )
        same = [pool[0], pool[0]]
        with self.assertRaisesRegex(ValueError, "unique"):
            m.select_candidates(same, "compression", 2)
        with self.assertRaisesRegex(ValueError, "exact"):
            m.select_candidates(pool, "compression", 1000)

    def test_boundary_categories_are_script_agnostic(self):
        self.assertEqual(m.boundary_kind("\u2581", "\u2581"), "whitespace")
        self.assertEqual(m.boundary_kind("\t", "\u2581"), "whitespace")
        self.assertEqual(m.boundary_kind("\u3002", "\u2581"), "punctuation")
        self.assertEqual(m.boundary_kind("\u093e", "\u2581"), "other")


if __name__ == "__main__":
    unittest.main()
