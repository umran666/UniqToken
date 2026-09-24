"""Exact packing tests use synthetic byte metadata, never research training."""

import ast
import copy
from itertools import product
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools import phase_b_exact_exposure as x
from tools import phase_b_exposure as b


def doc(identifier, size, domain="d", language="l"):
    return {
        "id": identifier,
        "domain": domain,
        "language": language,
        "normalized_text_hash": b.digest(identifier),
        "normalized_utf8_bytes": size,
        "source_utf8_bytes": size + 1,
        "source": {"fixture": identifier},
        "dedup": {"status": "fixture"},
        "train_row_index": 0,
    }


def all_strata():
    return [
        doc(f"{domain}:{language}:{suffix}", size, domain, language)
        for (domain, language), quota in b.quotas().items()
        for suffix, size in (("coverage", 2), ("rest", quota - 2))
    ]


def test_include_first_matches_exhaustive_enumeration():
    for n in range(5):
        for original in product(range(1, 5), repeat=n):
            for target in range(13):
                weights = [w for w in original if w <= target]
                expected = None
                for bits in product((True, False), repeat=len(weights)):
                    if sum(w for w, chosen in zip(weights, bits) if chosen) == target:
                        expected = [i for i, bit in enumerate(bits) if bit]
                        break
                actual, _ = x.subset_indices(weights, target)
                assert actual == expected, (weights, target)


def test_greedy_trap_distinct_equal_lengths_and_no_reuse():
    assert x.subset_indices([6, 5, 5], 10)[0] == [1, 2]
    assert x.subset_indices([5], 10)[0] is None
    assert x.subset_indices([2, 4, 6], 7)[0] is None
    assert x.subset_indices([], 0)[0] == []


def test_large_declared_dimensions_are_bounded():
    # Maximum declared n and T, but narrow reachable sets keep this regression
    # lightweight. The storage estimator still checks the full-width bound.
    chosen, report = x.subset_indices([1] * x.MAX_DOCUMENTS, x.MAX_TARGET)
    assert chosen is None
    assert report["boolean_state_bound"] == (x.MAX_DOCUMENTS + 1) * (x.MAX_TARGET + 1)
    assert report["estimated_snapshot_bytes"] <= x.MAX_SNAPSHOT_BYTES
    with pytest.raises(x.ResourceLimit):
        x.subset_indices([1] * x.MAX_DOCUMENTS, x.MAX_TARGET, snapshot_limit=1)


@pytest.mark.parametrize(
    "weights,target",
    [([0], 1), ([-1], 5), ([True], 5), ([10], 5), ([], -1), ([], x.MAX_TARGET + 1), ([1] * (x.MAX_DOCUMENTS + 1), 10)],
)
def test_invalid_solver_dimensions_fail(weights, target):
    with pytest.raises(ValueError):
        x.subset_indices(weights, target)


def test_coverage_cannot_be_swapped_to_make_solution():
    # 5 alone fills quota, but retaining mandatory shortest document 2 makes it impossible.
    chosen, report = x.solve_stratum([doc("short", 2), doc("long", 5)], 5)
    assert chosen is None and report["status"] == x.FAIL_NO_SOLUTION
    assert report["coverage_document_id"] == "short"
    chosen, report = x.solve_stratum([doc("too-large", 6)], 5)
    assert chosen is None and report["reason"] == "no_fitting_mandatory_coverage_document"


def test_tied_ranks_use_ids_and_coverage_size(monkeypatch):
    monkeypatch.setattr(b, "ranked", lambda d: {**d, "candidate_rank": "tied"})
    docs = [doc("c", 4), doc("b", 4), doc("a", 1)]
    chosen, report = x.solve_stratum(docs, 5)
    assert {d["id"] for d in chosen} == {"a", "b"}
    assert report["selected_ids_in_rank_order"] == ["a", "b"]


def test_all_30_exact_and_independent_verifier(monkeypatch):
    parent = all_strata()
    sample, reports = x.solve_all(parent, b.quotas())
    again, again_reports = x.solve_all(list(reversed(parent)), b.quotas())
    assert sample == again and reports == again_reports
    assert sample["normalized_utf8_bytes"] == 1_000_000
    assert len(reports) == 30 and all(r["status"] == "EXACT_SOLUTION" for r in reports)
    monkeypatch.setattr(b, "construct", lambda *a, **k: pytest.fail("verifier called constructor"))
    monkeypatch.setattr(x, "solve_all", lambda *a, **k: pytest.fail("verifier called solver"))
    assert set(x.verify_witness(parent, b.quotas(), sample).values()) == {"PASS"}


@pytest.mark.parametrize("mutation", ["duplicate", "hash_duplicate", "missing", "negative"])
def test_bad_parent_rejected(mutation):
    docs = all_strata()
    if mutation == "duplicate":
        docs.append(docs[0])
    elif mutation == "hash_duplicate":
        docs[1]["normalized_text_hash"] = docs[0]["normalized_text_hash"]
    elif mutation == "missing":
        docs = docs[2:]
    else:
        docs[0]["normalized_utf8_bytes"] = -1
    with pytest.raises(ValueError):
        x.solve_all(docs, b.quotas())


@pytest.mark.parametrize(
    "mutation",
    [
        "membership",
        "bytes",
        "source_bytes",
        "order",
        "rank",
        "coverage",
        "quota",
        "counts",
        "list_hash",
        "total",
        "source_total",
        "duplicate",
    ],
)
def test_independent_verifier_rejects_tampering(mutation):
    parent = all_strata()
    sample, _ = x.solve_all(parent, b.quotas())
    d = sample["ordered_documents"][0]
    if mutation == "membership":
        d["id"] = "not-in-parent"
    elif mutation == "bytes":
        d["normalized_utf8_bytes"] += 1
    elif mutation == "source_bytes":
        d["source_utf8_bytes"] += 1
    elif mutation == "order":
        sample["ordered_documents"][0], sample["ordered_documents"][1] = sample["ordered_documents"][1], d
    elif mutation == "rank":
        d["candidate_rank"] = "different"
    elif mutation == "coverage":
        d["coverage_document"] = False
    elif mutation == "quota":
        sample["strata"][0]["actual_normalized_bytes"] -= 1
    elif mutation == "counts":
        sample["strata"][0]["selected_documents"] += 1
    elif mutation == "list_hash":
        sample["strata"][0]["ordered_list_sha256"] = "changed"
    elif mutation == "total":
        sample["normalized_utf8_bytes"] -= 1
    elif mutation == "source_total":
        sample["source_utf8_bytes"] -= 1
    else:
        sample["ordered_documents"].append(copy.deepcopy(d))
    with pytest.raises(ValueError):
        x.verify_witness(parent, b.quotas(), sample)


def attempt_fixture(tmp_path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    policy = {"policy_id": x.POLICY}
    b.publish(attempt / "policy.json", policy)
    plan = {
        "timeout_seconds": 10,
        "input_hashes": {},
        "generator": {"fixture": True},
        "policy_sha256": b.file_hash(attempt / "policy.json"),
    }
    b.publish(attempt / "plan.json", plan)
    return attempt, plan


@pytest.mark.parametrize(
    "error,reason",
    [
        (subprocess.TimeoutExpired("worker", 1), "SEARCH_INCOMPLETE_RESOURCE_LIMIT"),
        (x.ResourceLimit("limit"), "SEARCH_INCOMPLETE_RESOURCE_LIMIT"),
        (MemoryError(), "SEARCH_INCOMPLETE_RESOURCE_LIMIT"),
        (KeyboardInterrupt(), "SEARCH_INTERRUPTED"),
        (SystemExit(), "SEARCH_INTERRUPTED"),
        (RuntimeError("native/worker error"), "SEARCH_ERROR"),
    ],
)
def test_interrupted_or_failed_search_seals_rejection(tmp_path, error, reason):
    attempt, plan = attempt_fixture(tmp_path)

    def child(*args):
        raise error

    receipt = x.supervise(attempt, plan, child)
    assert receipt["status"] == "exposure_rejected" and receipt["reason"] == reason
    assert receipt["final_exposure_sha256"] is None
    assert len(receipt["strata"]) == 30
    assert not (attempt / "frozen").exists()
    assert not (attempt / "exposure.json").exists()


def test_no_solution_cannot_invoke_verifier_or_publish(tmp_path):
    attempt, plan = attempt_fixture(tmp_path)
    calls = []

    def child(attempt, mode, deadline):
        calls.append(mode)
        b.publish(
            attempt / "search.json",
            {"all_exact": False, "strata": [{"domain": "d", "language": "l", "status": x.FAIL_NO_SOLUTION}]},
        )

    receipt = x.supervise(attempt, plan, child)
    assert receipt["reason"] == "NO_EXACT_SOLUTION"
    assert calls == ["solve"] and not (attempt / "frozen").exists()


def successful_child(attempt, mode, deadline):
    if mode == "solve":
        sample, reports = x.solve_all(all_strata(), b.quotas())
        b.publish(attempt / "search.json", {"all_exact": True, "strata": reports})
        b.publish(
            attempt / "candidate.json",
            {
                "sample": sample,
                "search": reports,
                "provenance": {"fixture": True, "source_manifest_sha256": b.SOURCE_SHA},
            },
        )
    else:
        candidate = b.read_json(attempt / "candidate.json")
        x.verify_witness(all_strata(), b.quotas(), candidate["sample"])
        b.publish(
            attempt / "verification.json",
            {"status": "PASS", "candidate_sha256": b.file_hash(attempt / "candidate.json")},
        )


def test_complete_only_atomic_bundle_publication(tmp_path):
    attempt, plan = attempt_fixture(tmp_path)
    result = x.supervise(attempt, plan, successful_child)
    assert result["status"] == "exact_packing_feasible"
    assert result["final_exposure_sha256"] == b.file_hash(attempt / "frozen/exposure.json")
    assert result["normalized_utf8_bytes"] == 1_000_000
    assert (attempt / "frozen/receipt.json").exists() and (attempt / "frozen/strata-report.json").exists()
    assert not (attempt / ".publication-incomplete").exists()
    manifest = b.read_json(attempt / "frozen/exposure.json")
    assert manifest["content_sha256"] == b.digest({k: v for k, v in manifest.items() if k != "content_sha256"})
    assert manifest["authorized_for_lm_execution"] is False


def test_failure_during_publication_never_exposes_bundle(tmp_path, monkeypatch):
    attempt, plan = attempt_fixture(tmp_path)
    monkeypatch.setattr(x.os, "rename", lambda *args: (_ for _ in ()).throw(OSError("publication failed")))
    receipt = x.supervise(attempt, plan, successful_child)
    assert receipt["reason"] == "SEARCH_ERROR" and not (attempt / "frozen").exists()


def test_stale_candidate_after_verification_rejected(tmp_path):
    attempt, plan = attempt_fixture(tmp_path)

    def child(attempt, mode, deadline):
        successful_child(attempt, mode, deadline)
        if mode == "verify":
            with (attempt / "candidate.json").open("a") as stream:
                stream.write(" ")

    result = x.supervise(attempt, plan, child)
    assert result["status"] == "exposure_rejected" and not (attempt / "frozen").exists()


@pytest.mark.parametrize("filename", ["plan.json", "policy.json"])
def test_changed_execution_plan_or_policy_rejected(tmp_path, filename):
    attempt, plan = attempt_fixture(tmp_path)

    def child(attempt, mode, deadline):
        successful_child(attempt, mode, deadline)
        if mode == "verify":
            (attempt / filename).write_text("{}")

    receipt = x.supervise(attempt, plan, child)
    assert receipt["status"] == "exposure_rejected" and not (attempt / "frozen").exists()


def test_failed_setup_leaves_rejected_receipt_and_refuses_reuse(tmp_path):
    policy = tmp_path / "policy.md"
    policy.write_text("not approved")
    args = SimpleNamespace(output=tmp_path / "attempt", policy=policy)
    with pytest.raises(ValueError, match="unapproved"):
        x.generate(args)
    initial = args.output / "initial-rejection-receipt.json"
    assert b.read_json(initial)["final_exposure_sha256"] is None
    assert (args.output / "setup-rejection-receipt.json").exists()
    before = initial.read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        x.generate(args)
    assert initial.read_bytes() == before


def test_worker_terminated_by_timeout_is_killed_and_waited(tmp_path, monkeypatch):
    class Child:
        killed = False
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            return 1

        def kill(self):
            self.killed = True

    child = Child()
    monkeypatch.setattr(x.subprocess, "Popen", lambda *a, **k: child)
    with pytest.raises(subprocess.TimeoutExpired):
        x.run_child(tmp_path, "solve", x.time.monotonic() + 1)
    assert child.killed and child.calls == 2


def test_no_tokenizer_flop_or_lm_dependencies():
    tree = ast.parse(Path(x.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module)
    assert modules <= {
        "__future__",
        "argparse",
        "collections",
        "json",
        "os",
        "pathlib",
        "signal",
        "subprocess",
        "sys",
        "time",
        "tools",
    }
