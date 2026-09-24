from __future__ import annotations

import copy

import pytest

from benchmarks import run_research_experiments as research
from tools import phase_c_exposure as gate


def doc(identifier, index, size, domain="domain", language="lang"):
    return {
        "id": identifier,
        "domain": domain,
        "language": language,
        "train_row_index": index,
        "normalized_text_hash": f"hash-{identifier}",
        "normalized_utf8_bytes": size,
        "source_utf8_bytes": size + 1,
        "source": {"dataset": "source"},
        "dedup": {"status": "accepted_after_exact_and_near_eval_check", "truncated_to_quota": False},
        "upstream_text_sha256": f"upstream-{identifier}",
    }


def test_largest_remainder_is_exact_and_ties_use_stratum_order():
    source = {("b", "x"): 1, ("a", "x"): 1, ("c", "x"): 1}
    assert gate.proportional_quotas(source, 2) == {
        ("a", "x"): 1,
        ("b", "x"): 1,
        ("c", "x"): 0,
    }


def test_exact_tail_preserves_source_order_and_whole_documents(monkeypatch):
    monkeypatch.setattr(gate, "MAX_TARGET", 10)
    rows = [doc("a", 0, 7), doc("b", 1, 6), doc("c", 2, 4), doc("d", 3, 3)]
    selected, report = gate.select_stratum(rows, 17)
    assert sum(row["normalized_utf8_bytes"] for row in selected) == 17
    assert [row["train_row_index"] for row in selected] == sorted(row["train_row_index"] for row in selected)
    assert report["tail_target"] <= 10


def test_exact_tail_search_bound_uses_first_candidates_in_source_order(monkeypatch):
    monkeypatch.setattr(gate, "MAX_TARGET", 10)
    monkeypatch.setattr(gate, "MAX_DOCUMENTS", 2)
    rows = [doc("a", 0, 5), doc("b", 1, 3), doc("c", 2, 2), doc("d", 3, 1)]
    selected, report = gate.select_stratum(rows, 5)
    assert [row["id"] for row in selected] == ["a"]
    assert report["eligible_tail_documents"] == 4
    assert report["searched_tail_documents"] == 2


def test_phase_b_documents_are_excluded(monkeypatch):
    monkeypatch.setattr(gate, "EXACT_BYTES", 1)
    rows = [doc("old", 0, 1), doc("new", 1, 1)]
    selected, reports = gate.select_all(rows, {("domain", "lang"): 1}, {"old"})
    assert [row["id"] for row in selected] == ["new"]
    assert reports[0]["selected_normalized_bytes"] == 1


def test_independent_verifier_rejects_altered_witness(monkeypatch):
    rows = [doc("a", 0, 1)]
    monkeypatch.setattr(gate, "EXACT_BYTES", 1)
    monkeypatch.setattr(gate, "load_training", lambda *_: (rows, None))
    exposure = gate.exposure_body(
        rows, [{"domain": "domain", "language": "lang"}], {("domain", "lang"): 1}, {"source_manifest_sha256": "x"}
    )
    assert gate.verify_exposure(None, {}, exposure, set())["status"] == "PASS"
    changed = copy.deepcopy(exposure)
    changed["ordered_documents"][0]["normalized_utf8_bytes"] = 2
    with pytest.raises(ValueError, match="altered"):
        gate.verify_exposure(None, {}, changed, set())


def test_validation_assignment_uses_confirmation_half_and_never_test(monkeypatch):
    rows = [{"raw_utf8_bytes": 1}, {"raw_utf8_bytes": 1}]
    texts = ["a", "b"]
    monkeypatch.setattr(
        gate.stages,
        "_load_rows",
        lambda path, manifest, split: (rows, texts) if split == "validation" else pytest.fail("test opened"),
    )
    monkeypatch.setattr(gate.stages, "partition_validation", lambda r, t: ([(r[0], t[0])], [(r[1], t[1])]))
    result = gate.validation_assignment(None, {}, set())
    assert result["partition"] == "confirmation"
    assert result["ordered_normalized_text_hashes"] == [gate.base.digest("b")]
    assert result["test_split_opened"] is False


def test_validation_overlap_is_rejected(monkeypatch):
    rows = [{"raw_utf8_bytes": 1}, {"raw_utf8_bytes": 1}]
    texts = ["a", "b"]
    monkeypatch.setattr(gate.stages, "_load_rows", lambda *_: (rows, texts))
    monkeypatch.setattr(gate.stages, "partition_validation", lambda r, t: ([(r[0], t[0])], [(r[1], t[1])]))
    with pytest.raises(ValueError, match="overlap"):
        gate.validation_assignment(None, {}, {gate.base.digest("b")})


def test_exposure_content_hash_detects_tampering(monkeypatch):
    rows = [doc("a", 0, 1)]
    monkeypatch.setattr(gate, "EXACT_BYTES", 1)
    monkeypatch.setattr(gate, "load_training", lambda *_: (rows, None))
    exposure = gate.exposure_body(rows, [], {("domain", "lang"): 1}, {})
    exposure["source_utf8_bytes"] += 1
    with pytest.raises(ValueError, match="source-byte"):
        gate.verify_exposure(None, {}, exposure, set())
