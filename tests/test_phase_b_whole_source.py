"""Whole-record regeneration must preserve upstream text and exact quotas."""
import copy

import pytest

from tools import phase_b_whole_source as w


def record(text, truncated=False):
    return {"id": text, "text": text, "raw_utf8_bytes": len(text.encode()),
            "normalized_utf8_bytes": len(w.base.normalize(text).encode()),
            "dedup": {"truncated_to_quota": truncated, "status": "old"}}


def test_restore_prefix_preserves_complete_upstream_and_evidence():
    old = record("ab", True)
    before = copy.deepcopy(old)
    new = w.restored_record(old, "abcdef")
    assert old == before
    assert new["text"] == "abcdef" and new["raw_utf8_bytes"] == 6
    assert new["dedup"]["truncated_to_quota"] is False
    assert new["dedup"]["status"] == "pending_regenerated_corpus_dedup"
    assert new["whole_record_provenance"]["original_was_truncated"] is True


@pytest.mark.parametrize("old,text", [(record("ab", True), "xyz"), (record("ab"), "abcdef")])
def test_wrong_upstream_rejected(old, text):
    with pytest.raises(ValueError, match="upstream text"):
        w.restored_record(old, text)


def test_exact_removal_retains_whole_documents_in_original_order():
    rows = [record("a" * 6), record("b" * 4), record("c" * 3)]
    kept, removed = w.select_exact(rows, 9)
    assert kept == [rows[0], rows[2]]
    assert removed == [rows[1]["id"]]
    assert sum(r["normalized_utf8_bytes"] for r in kept) == 9


def test_impossible_exact_selection_rejects_instead_of_truncation():
    with pytest.raises(ValueError, match="no exact whole-record"):
        w.select_exact([record("abcd"), record("efgh")], 7)


def test_insufficient_candidate_pool_rejected():
    with pytest.raises(ValueError, match="below quota"):
        w.select_exact([record("a")], 2)


def test_normalized_and_source_bytes_reported_separately():
    new = w.restored_record(record("\uff21"), "\uff21")
    assert new["raw_utf8_bytes"] == 3
    assert new["normalized_utf8_bytes"] == 1


def test_gzip_reader_uses_exact_record_index(tmp_path):
    import gzip
    import json
    source = tmp_path / "source.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        for text in ["first", "second", "third"]:
            stream.write(json.dumps({"text": text}) + "\n")
    assert list(w.upstream_rows(source, {1})) == [(1, "second")]


def test_resource_bound_does_not_relax_quota():
    with pytest.raises(ValueError, match="declared bound"):
        w.select_exact([record("a" * (w.MAX_TARGET + 2))], 1)


def test_tail_can_fill_exact_quota_with_additional_whole_source(monkeypatch):
    monkeypatch.setattr(w, "MAX_TARGET", 8)
    rows = [w.restored_record(record(s), s) for s in ["aaaaaa", "bbbb"]]
    extra = w.restored_record(record("ccc"), "ccc")
    chosen, _ = w.select_whole_tail(rows, 9, lambda: iter([extra]))
    assert [r["text"] for r in chosen] == ["aaaaaa", "ccc"]


def test_tail_excludes_exact_duplicate_candidates():
    row = w.restored_record(record("aaa"), "aaa")
    with pytest.raises(ValueError, match="no exact whole-record tail"):
        w.select_whole_tail([row], 6, lambda: iter([copy.deepcopy(row)]))


@pytest.fixture
def regeneration_lineage(tmp_path):
    source = {
        "dataset_id": "fixture", "normalization": "NFKC_unicode_spaces_v1",
        "freeze": {"source_revisions": {"dataset": "frozen"}},
        "splits": {"validation": {"sha256": "v"}, "test": {"sha256": "t"}},
        "whole_record_regeneration": {
            "policy": w.POLICY, "historical_source_manifest_sha256": w.base.SOURCE_SHA,
            "unchanged_selection_sha256": w.base.SELECTION_SHA,
            "all_selected_text_verified_against_upstream": True,
            "deduplication": "passed_existing_exact_and_minhash_lsh_train_evaluation_checks",
        },
    }
    selection = {"dataset": {"dataset_id": "fixture", "normalization": source["normalization"],
                  "source_revisions": source["freeze"]["source_revisions"],
                  "validation_file_sha256": "v", "untouched_test_file_sha256": "t"}}
    path = tmp_path / "manifest.json"
    w.base.publish(path, source)
    w.base.publish(tmp_path / "receipt.json", {"status": "whole_upstream_source_frozen",
                   "manifest_sha256": w.base.file_hash(path)})
    return source, path, selection


def test_explicit_regeneration_lineage_preserves_selection(regeneration_lineage):
    source, path, selection = regeneration_lineage
    before = copy.deepcopy(selection)
    w.base.validate_regenerated_source(source, path, selection)
    assert selection == before


@pytest.mark.parametrize("field", ["policy", "historical_source_manifest_sha256",
                                  "unchanged_selection_sha256", "deduplication",
                                  "all_selected_text_verified_against_upstream"])
def test_invalid_regeneration_lineage_rejected(regeneration_lineage, field):
    source, path, selection = regeneration_lineage
    source["whole_record_regeneration"][field] = "changed"
    with pytest.raises(ValueError):
        w.base.validate_regenerated_source(source, path, selection)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_regeneration_cannot_change_evaluation(regeneration_lineage, split):
    source, path, selection = regeneration_lineage
    source["splits"][split]["sha256"] = "changed"
    with pytest.raises(ValueError, match="evaluation split"):
        w.base.validate_regenerated_source(source, path, selection)


def test_changed_manifest_receipt_rejected(regeneration_lineage):
    source, path, selection = regeneration_lineage
    with path.open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="unverified regenerated"):
        w.base.validate_regenerated_source(source, path, selection)
