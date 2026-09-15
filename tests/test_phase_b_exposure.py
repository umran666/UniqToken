"""Synthetic byte-only exposure checks; no experiments or dataset sampling."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import phase_b_exposure as e


def doc(domain, language, identifier, size):
    return {"domain": domain, "language": language, "id": identifier,
            "normalized_text_hash": e.digest(identifier), "normalized_utf8_bytes": size,
            "source_utf8_bytes": size + 2, "train_row_index": 0, "source": {}, "dedup": {}}


def test_all_thirty_allocations_match_review_table():
    allocations = e.quotas()
    assert len(allocations) == 30 and sum(allocations.values()) == 1_000_000
    table = (e.ROOT / "benchmarks/PHASE_B_EXPOSURE_POLICY_REVIEW.md").read_text(encoding="utf-8")
    actual = {}
    for line in table.splitlines():
        if line.startswith("| ") and " / " in line and "Pending" in line:
            cells = [cell.strip() for cell in line.split("|")]
            domain, language = cells[1].split(" / ")
            language = language.split(" ")[0]
            actual[domain, language] = int(cells[2])
            assert float(cells[3].rstrip("%")) == int(cells[2]) / 10000
            assert int(cells[4].split(" / ")[1]) == (int(cells[2]) * 9 + 9) // 10
    assert allocations == actual


def test_every_stratum_coverage_and_determinism():
    documents = []
    for (domain, language), quota in e.quotas().items():
        documents.extend([doc(domain, language, f"{domain}:{language}:short", 2),
                          doc(domain, language, f"{domain}:{language}:rest", quota - 2)])
    first = e.construct(documents, e.quotas())
    assert first == e.construct(list(reversed(documents)), e.quotas())
    assert first["normalized_utf8_bytes"] == 1_000_000
    assert first["selected_documents"] == 60
    ordered = first["ordered_documents"]
    assert all(d["coverage_document"] for d in ordered[:30])
    assert not any(d["coverage_document"] for d in ordered[30:])
    assert len({d["domain"] for d in ordered[:4]}) == 4
    assert len({(d["domain"], d["language"]) for d in ordered[:30]}) == 30
    assert {s["packing_status"] for s in first["strata"]} == {"PASS"}
    assert {s["exact_quota_status"] for s in first["strata"]} == {"PASS"}
    for stratum in first["strata"]:
        selected = [ordered[i] for i in stratum["global_positions"]]
        assert stratum["ordered_document_ids"] == [d["id"] for d in selected]
        assert stratum["source_utf8_bytes"] == sum(d["source_utf8_bytes"] for d in selected)


def test_rank_ties_and_weighted_service_are_explicit(monkeypatch):
    monkeypatch.setattr(e, "digest", lambda value: "rank")
    docs = [doc("d", "b", "b2", 3), doc("d", "a", "a1", 1), doc("d", "b", "b1", 1), doc("d", "a", "a2", 3)]
    # Text hashes cannot collide even though the rank hash is deliberately tied.
    for d in docs:
        d["normalized_text_hash"] = d["id"]
    result = e.construct(docs, {("d", "a"): 4, ("d", "b"): 4})
    assert [d["id"] for d in result["ordered_documents"]] == ["a1", "b1", "a2", "b2"]


@pytest.mark.parametrize("sizes,quota,reason,status", [([4, 10], 10, "no_whole_document_fits", "FAIL"),
                                                      ([9], 10, "eligible_pool_exhausted", "PASS"),
                                                      ([10], 10, "quota_filled", "PASS"),
                                                      ([11], 10, "no_whole_document_fits", "FAIL")])
def test_whole_documents_and_exhaustion(sizes, quota, reason, status):
    documents = [doc("d", "l", str(i), size) for i, size in enumerate(sizes)]
    result = e.construct(documents, {("d", "l"): quota})
    row = result["strata"][0]
    assert row["exhaustion_reason"] == reason and row["packing_status"] == status
    assert result["normalized_utf8_bytes"] <= quota
    assert all(d["normalized_utf8_bytes"] in sizes for d in result["ordered_documents"])


@pytest.mark.parametrize("mutation", ["id", "text", "stratum", "length"])
def test_duplicate_unapproved_and_invalid_candidates_fail(mutation):
    docs = [doc("d", "l", "one", 2), doc("d", "l", "two", 3)]
    if mutation == "id":
        docs[1]["id"] = docs[0]["id"]
    elif mutation == "text":
        docs[1]["normalized_text_hash"] = docs[0]["normalized_text_hash"]
    elif mutation == "stratum":
        docs[1]["language"] = "other"
    else:
        docs[1]["normalized_utf8_bytes"] = 0
    with pytest.raises(ValueError):
        e.construct(docs, {("d", "l"): 5})


def test_atomic_failed_attempt_is_not_a_manifest(tmp_path):
    sample = e.construct([doc("d", "l", "short", 9)], {("d", "l"): 10})
    # Meets the earlier 90% gate but must fail the latest exact-budget requirement.
    output = tmp_path / "attempt"
    receipt = e.persist_attempt(output, {"required_normalized_bytes": 10},
                                {"selection_sha256": "s", "source_manifest_sha256": "m"}, sample, {})
    assert receipt["status"] == "exposure_rejected" and receipt["final_exposure_sha256"] is None
    assert receipt["tokenizer_preflight_run"] is False
    assert not (output / "exposure.json").exists()
    rejected = output / "rejected-exposure.json"
    assert receipt["artifact_sha256"] == e.file_hash(rejected)
    before = rejected.read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        e.persist_attempt(output, {}, {}, sample, {})
    assert rejected.read_bytes() == before


def test_success_publication_hash_chain(tmp_path):
    sample = e.construct([doc("d", "l", "complete", 10)], {("d", "l"): 10})
    output = tmp_path / "success"
    receipt = e.persist_attempt(output, {"required_normalized_bytes": 10},
                                {"selection_sha256": "s", "source_manifest_sha256": "m"}, sample, {})
    manifest = e.read_json(output / "exposure.json")
    assert receipt["status"] == "exposure_frozen"
    assert receipt["final_exposure_sha256"] == e.file_hash(output / "exposure.json")
    assert manifest["policy_sha256"] == e.file_hash(output / "policy.json")
    assert manifest["content_sha256"] == e.digest({k: v for k, v in manifest.items() if k != "content_sha256"})
    assert e.read_json(output / "strata-report.json")["manifest_sha256"] == receipt["final_exposure_sha256"]


def parent_fixture(tmp_path):
    path = tmp_path / "train.jsonl"
    source_item = {"local_path": "sources/fixture", "sha256": "h", "dataset": "fixture",
                   "revision": "r", "release_variant": "v", "url": "pinned", "license": "test"}
    rows = []
    groups = []
    for i, key in enumerate(e.quotas()):
        text = f"document-{i}-\ufb01"
        size = len(e.normalize(text).encode("utf-8"))
        rows.append({"id": str(i), "domain": key[0], "language": key[1], "text": text,
                     "raw_utf8_bytes": len(text.encode("utf-8")), "normalized_utf8_bytes": size,
                     "dedup": {"status": "accepted_after_exact_and_near_eval_check"},
                     "source": {**source_item, "source_file_sha256": "h"}})
        groups.append({"domain": key[0], "language": key[1], "target_bytes": size,
                       "actual_bytes": size, "documents": 1})
    path.write_text("".join(e.json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    source = {"splits": {"train": {"path": path.name, "sha256": e.file_hash(path)},
                         "validation": {"path": "does-not-exist"}, "test": {"path": "does-not-exist"}},
              "freeze": {"source_files": [source_item]}}
    training = {"assignment_hash": e.digest([e.digest(e.normalize(r["text"])) for r in rows]),
                "documents": len(rows), "normalized_utf8_bytes": sum(r["normalized_utf8_bytes"] for r in rows),
                "source_utf8_bytes": sum(r["raw_utf8_bytes"] for r in rows), "screen_selection": {"groups": groups}}
    return source, training


def test_source_loader_opens_only_training_and_verifies_parent(tmp_path):
    source, training = parent_fixture(tmp_path)
    docs, _ = e.load_parent(tmp_path / "manifest.json", source, training)
    assert len(docs) == 30
    assert sum(d["normalized_utf8_bytes"] for d in docs) == training["normalized_utf8_bytes"]
    assert docs[0]["source_utf8_bytes"] > docs[0]["normalized_utf8_bytes"]


def test_per_document_license_can_differ_from_inventory_summary(tmp_path):
    source, training = parent_fixture(tmp_path)
    source["freeze"]["source_files"][0]["license"] = "permissive licenses recorded per document"
    docs, _ = e.load_parent(tmp_path / "manifest.json", source, training)
    assert all(d["source"]["license"] == "test" for d in docs)


@pytest.mark.parametrize("mutation", ["train_hash", "assignment", "group", "source", "bytes"])
def test_stale_parent_provenance_rejected(tmp_path, mutation):
    source, training = parent_fixture(tmp_path)
    if mutation == "train_hash":
        source["splits"]["train"]["sha256"] = "wrong"
    elif mutation == "assignment":
        training["assignment_hash"] = "wrong"
    elif mutation == "group":
        training["screen_selection"]["groups"][0]["actual_bytes"] -= 1
    elif mutation == "source":
        source["freeze"]["source_files"][0]["sha256"] = "wrong"
    else:
        training["source_utf8_bytes"] -= 1
    with pytest.raises(ValueError):
        e.load_parent(tmp_path / "manifest.json", source, training)


def test_sampler_is_standard_library_only_and_has_no_preflight_api():
    code = Path(e.__file__).read_text(encoding="utf-8")
    tree = ast.parse(code)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "argparse", "collections", "copy", "fractions", "hashlib", "json", "os", "pathlib",
                        "re", "subprocess", "tempfile", "unicodedata"}
    assert not hasattr(e, "preflight") and not hasattr(e, "encode")
    # Adding unrelated score fields cannot affect candidate choice/order.
    docs = [doc("d", "l", "one", 2), doc("d", "l", "two", 4)]
    original = e.construct(docs, {("d", "l"): 5})
    changed = copy.deepcopy(docs)
    for d in changed:
        d["tokenizer_score"] = -100
    tested = e.construct(changed, {("d", "l"): 5})
    assert [d["id"] for d in original["ordered_documents"]] == [d["id"] for d in tested["ordered_documents"]]


def test_frozen_selection_and_dataset_hash_rejected_before_reading(tmp_path):
    selection = tmp_path / "selection.json"
    source = tmp_path / "source.json"
    selection.write_text("{}")
    source.write_text("{}")
    with pytest.raises(ValueError, match="selection SHA"):
        e.input_context(selection, source)


def test_atomic_publication_refuses_overwrite(tmp_path):
    path = tmp_path / "evidence.json"
    e.publish(path, {"original": True})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        e.publish(path, {"original": False})
    assert path.read_bytes() == before
