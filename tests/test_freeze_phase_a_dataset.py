"""Unit tests for the Phase A corpus freezer; no network or research data required."""

import json
from pathlib import Path

import pytest

from benchmarks import freeze_phase_a_dataset as freeze
from benchmarks.run_research_experiments import file_hash, normalize


def source(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "source.txt"
    path.write_text("immutable source", encoding="utf-8")
    return {
        "dataset": "example/pinned",
        "revision": "a" * 40,
        "url": "https://example.invalid/pinned/a/source.txt",
        "remote_path": "source.txt",
        "local_path": path.name,
        "local_path_abs": str(path),
        "sha256": file_hash(path),
        "license": "MIT",
    }


def test_phase_a_quotas_are_decimal_and_exact():
    assert freeze.MB == 1_000_000
    assert sum(total for total, _ in freeze.PROSE_STRATA.values()) == 400_000_000
    assert 100 * freeze.MB == 100_000_000
    for total, languages in freeze.PROSE_STRATA.values():
        assert sum(freeze.quota_by_language(total, languages).values()) == total


def test_the_stack_allowlist_is_explicitly_permissive():
    assert {"mit", "apache-2.0", "bsd-3-clause"} <= freeze.PERMISSIVE_STACK_LICENSES
    assert not {"gpl-3.0", "proprietary", "unknown"} & freeze.PERMISSIVE_STACK_LICENSES


@pytest.mark.parametrize("revision", ["main", "v1.3", "a" * 39, "A" * 40])
def test_fixed_revision_requires_lowercase_commit_hash(revision):
    with pytest.raises(ValueError):
        freeze.fixed_revision(revision)


def test_normalized_prefix_never_breaks_utf8_or_misses_quota():
    text = "ab\u00a0\u4e2d\u6587"
    chosen = freeze.normalized_prefix(text, len(normalize("ab ").encode("utf-8")))
    assert chosen == "ab\u00a0"
    assert len(normalize(chosen).encode("utf-8")) == 3
    assert freeze.normalized_prefix("\u4e2d", 1) is None


def test_append_exact_records_provenance_and_byte_fields(tmp_path):
    output = tmp_path / "selected.jsonl"
    stats = freeze.append_exact(
        [("abc def", source(tmp_path), 7)], 4, output, language="en", domain="latin_english"
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert stats == {"documents": 1, "normalized_utf8_bytes": 4}
    assert row["text"] == "abc "
    assert row["raw_utf8_bytes"] == 4
    assert row["normalized_utf8_bytes"] == 4
    assert row["source"]["source_file_sha256"] == source(tmp_path)["sha256"]
    assert row["dedup"]["status"] == "accepted_after_exact_and_near_eval_check"


def write_rows(path: Path, texts: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts), encoding="utf-8"
    )


def test_dedup_rejects_exact_and_near_training_duplicates(tmp_path):
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    write_rows(train, ["a" * 100, "a" * 100])
    write_rows(evaluation, ["unrelated evaluation text"])
    with pytest.raises(ValueError, match="duplicate normalized training document"):
        freeze.validate_dedup(train, [evaluation])

    base = " ".join(f"token{index:03d}" for index in range(100))
    write_rows(train, [base + "x", base + "y"])
    with pytest.raises(ValueError, match="near duplicate training document"):
        freeze.validate_dedup(train, [evaluation])


def test_dedup_rejects_train_evaluation_near_overlap(tmp_path):
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    base = " ".join(f"token{index:03d}" for index in range(100))
    write_rows(train, [base + "x"])
    write_rows(evaluation, [base + "y"])
    with pytest.raises(ValueError, match="near train/evaluation overlap invalidates manifest"):
        freeze.validate_dedup(train, [evaluation])
