"""Unit tests for the Phase A corpus freezer; no network or research data required."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

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
        "release_variant": "test-v1",
        "file_bytes": path.stat().st_size,
    }


def test_phase_a_quotas_are_decimal_and_exact():
    assert freeze.MB == 1_000_000
    assert sum(total for total, _ in freeze.PROSE_STRATA.values()) == 400_000_000
    assert 100 * freeze.MB == 100_000_000
    for total, languages in freeze.PROSE_STRATA.values():
        assert sum(freeze.quota_by_language(total, languages).values()) == total


def test_the_stack_cpp_source_directory_matches_upstream():
    assert freeze.CODE_SOURCE_DIRECTORIES["cpp"] == "c++"


def test_the_stack_license_allowlist_is_loaded_from_pinned_source(tmp_path):
    path = tmp_path / "licenses.json"
    path.write_text('["MIT", "Apache-2.0", "BSD-3-Clause"]', encoding="utf-8")
    assert freeze.stack_license_allowlist({"local_path_abs": str(path)}) == {
        "mit",
        "apache-2.0",
        "bsd-3-clause",
    }


def test_the_stack_reader_uses_v13_license_columns_and_filters_non_allowlisted_rows(tmp_path):
    path = tmp_path / "stack.parquet"
    pq.write_table(
        pa.table(
            {
                "content": ["mit code", "gpl code", "missing license"],
                "max_stars_repo_licenses": [["MIT"], ["GPL-3.0"], None],
                "max_issues_repo_licenses": [None, None, None],
                "max_forks_repo_licenses": [None, None, None],
            }
        ),
        path,
    )
    rows = list(
        freeze.stack_records(
            {"local_path_abs": str(path), "license": freeze.STACK_LICENSE},
            frozenset({"mit", "apache-2.0"}),
        )
    )
    assert [(text, row_source["license"], index) for text, row_source, index in rows] == [
        ("mit code", "mit", 0)
    ]


def valid_selection_groups():
    groups = []
    for domain, (total, languages) in freeze.PROSE_STRATA.items():
        for language, size in freeze.quota_by_language(total, languages).items():
            groups.append(
                {
                    "split": "train",
                    "dataset": freeze.MADLAD,
                    "release_variant": "data-v1p5/clean_docs_v2",
                    "language": language,
                    "domain": domain,
                    "documents": 1,
                    "raw_utf8_bytes": size,
                    "normalized_utf8_bytes": size,
                }
            )
    for language, size in freeze.quota_by_language(100 * freeze.MB, freeze.CODE_LANGUAGES).items():
        groups.append(
            {
                "split": "train",
                "dataset": freeze.STACK,
                "release_variant": freeze.STACK_RELEASE,
                "language": language,
                "domain": "code",
                "documents": 1,
                "raw_utf8_bytes": size,
                "normalized_utf8_bytes": size,
            }
        )
    for split in ("validation", "test"):
        for language in freeze.FLORES_LANGUAGE_FILES:
            groups.append(
                {
                    "split": split,
                    "dataset": "facebook/flores",
                    "release_variant": "FLORES-200/all",
                    "language": language,
                    "domain": "flores200",
                    "documents": 1,
                    "raw_utf8_bytes": 1,
                    "normalized_utf8_bytes": 1,
                }
            )
    return groups


def test_selection_gate_enforces_quotas_and_language_coverage():
    groups = valid_selection_groups()
    freeze.validate_selection(groups)
    groups[0]["normalized_utf8_bytes"] -= 1
    with pytest.raises(ValueError, match="normalized-byte quota"):
        freeze.validate_selection(groups)


def test_repo_files_filters_clean_madlad_variant():
    class FakeApi:
        def list_repo_tree(self, *args, **kwargs):
            return [
                SimpleNamespace(path="data-v1p5/en/clean_docs_v2-00001.jsonl.gz", size=12),
                SimpleNamespace(path="data-v1p5/en/noisy_docs_v2-00001.jsonl.gz", size=15),
            ]

    assert freeze.repo_files(
        FakeApi(), "example/data", "a" * 40, "data-v1p5/en", filename_prefix="clean_docs_v2-"
    ) == ["data-v1p5/en/clean_docs_v2-00001.jsonl.gz"]


def test_source_preflight_checks_metadata_without_downloading(monkeypatch):
    calls = []

    def metadata(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(size=123)

    monkeypatch.setattr(freeze, "get_hf_file_metadata", metadata)
    freeze.preflight_source_access("example/data", "a" * 40, "data/file.parquet", "token")
    assert calls[0][0].endswith("/datasets/example/data/resolve/" + "a" * 40 + "/data/file.parquet")
    assert calls[0][1]["token"] == "token"


def test_flores_all_parquet_extracts_required_languages(tmp_path):
    path = tmp_path / "dev.parquet"
    columns = {"id": pa.array([17])}
    columns.update({f"sentence_{code}": pa.array([f"text-{code}"]) for code in freeze.FLORES_LANGUAGE_FILES.values()})
    pq.write_table(pa.table(columns), path)
    rows = list(freeze.flores_records({"local_path_abs": str(path)}))
    assert len(rows) == len(freeze.FLORES_LANGUAGE_FILES)
    assert ("hi", "text-hin_Deva", 17) in rows
    assert ("am", "text-amh_Ethi", 17) in rows


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


def test_normalized_prefix_is_bounded_for_large_final_document(monkeypatch):
    calls = 0
    original = freeze.normalize

    def counted(text):
        nonlocal calls
        calls += 1
        return original(text)

    monkeypatch.setattr(freeze, "normalize", counted)
    chosen = freeze.normalized_prefix("a" * 2_000_000, 390_130)
    assert chosen == "a" * 390_130
    assert calls < 100


def test_append_exact_records_provenance_and_byte_fields(tmp_path):
    output = tmp_path / "selected.jsonl"
    stats = freeze.append_exact(
        [("abc def", source(tmp_path), 7)], 4, output, language="en", domain="latin_english"
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert stats == {"documents": 1, "raw_utf8_bytes": 4, "normalized_utf8_bytes": 4}
    assert row["text"] == "abc "
    assert row["raw_utf8_bytes"] == 4
    assert row["normalized_utf8_bytes"] == 4
    assert row["source"]["source_file_sha256"] == source(tmp_path)["sha256"]
    assert row["dedup"]["status"] == "accepted_after_exact_and_near_eval_check"


def test_training_dedup_replaces_duplicate_across_partitions(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    pinned = source(tmp_path)
    index = freeze.DedupIndex()
    freeze.append_exact(
        [("aaaa", pinned, 0)],
        4,
        first,
        language="en",
        domain="latin_english",
        dedup_index=index,
    )
    freeze.append_exact(
        [("aaaa", pinned, 0), ("bbbb", pinned, 1)],
        4,
        second,
        language="python",
        domain="code",
        dedup_index=index,
    )
    assert json.loads(second.read_text(encoding="utf-8"))["text"] == "bbbb"


def test_resumed_partition_restores_cross_partition_dedup_state(tmp_path):
    first = tmp_path / "first.jsonl"
    pinned = source(tmp_path)
    freeze.append_exact([("unique", pinned, 0)], 6, first, language="en", domain="latin_english")
    index = freeze.DedupIndex()
    freeze.restore_dedup_index(first, index)
    assert not index.accept("unique")


def test_completed_partition_and_sources_can_resume(tmp_path):
    work = tmp_path / "freeze.partial-test"
    work.mkdir()
    pinned = source(work)
    part = work / "train-prose-en.jsonl"
    freeze.append_exact([("abcd", pinned, 0)], 4, part, language="en", domain="latin_english")
    assert freeze.completed_part(part, 4, language="en", domain="latin_english")
    resumed = freeze.resume_sources(part, work)
    assert len(resumed) == 1
    assert resumed[0]["sha256"] == pinned["sha256"]
    assert resumed[0]["local_path_abs"] == str(work / pinned["local_path"])


def test_copy_source_reuses_existing_pinned_file(tmp_path, monkeypatch):
    work = tmp_path / "freeze.partial-test"
    local = work / "sources" / "example--data" / ("a" * 40) / "data" / "source.txt"
    local.parent.mkdir(parents=True)
    local.write_text("already downloaded", encoding="utf-8")
    monkeypatch.setattr(
        freeze,
        "hf_hub_download",
        lambda *args, **kwargs: pytest.fail("existing source must not be downloaded again"),
    )
    result = freeze.copy_source(
        "example/data", "a" * 40, "data/source.txt", work, "MIT", "v1", token=None
    )
    assert result["sha256"] == file_hash(local)
    assert result["file_bytes"] == local.stat().st_size


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
