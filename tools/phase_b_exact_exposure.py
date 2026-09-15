"""Exact tokenizer-blind packing gate. No FLOP, tokenizer or LM execution API.

Run as python -m tools.phase_b_exact_exposure generate (...). A supervisor keeps
rejection evidence even when the separate packing/verifying process is killed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools import phase_b_exposure as base

POLICY = "phase_b_stratified_whole_document_exact_subset_v2"
REVIEW_SHA = "9df4fb5a2f4117f65cb398d92aefe476c5a0edf7f26238f513266a113f228e9a"
MAX_SECONDS = 300
MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
MAX_DOCUMENTS = 10931
MAX_TARGET = 320000
FAIL_NO_SOLUTION = "INFEASIBLE_UNDER_FIXED_COVERAGE_AND_QUOTA"


class ResourceLimit(RuntimeError):
    pass


def subset_indices(weights, target, snapshot_limit=MAX_SNAPSHOT_BYTES):
    """Suffix reachable sums and include-first reconstruction; zero-one, not reuse."""
    base.require(type(target) is int and 0 <= target <= MAX_TARGET, "invalid target")
    base.require(len(weights) <= MAX_DOCUMENTS and all(type(w) is int and 0 < w <= target for w in weights),
                 "invalid bounded candidate lengths")
    # Include integer object headers, list pointers and transient full-width values.
    integer_bytes = sys.getsizeof(0) + ((target + sys.int_info.bits_per_digit) // sys.int_info.bits_per_digit) * sys.int_info.sizeof_digit
    estimate = (len(weights) + 5) * integer_bytes + (len(weights) + 1) * 8 + 64
    dimensions = {"candidate_count": len(weights), "residual_target": target,
                  "boolean_state_bound": (len(weights) + 1) * (target + 1),
                  "estimated_snapshot_bytes": estimate, "snapshot_limit_bytes": snapshot_limit}
    if estimate > snapshot_limit:
        raise ResourceLimit(f"snapshot storage bound {estimate} exceeds {snapshot_limit}")
    mask = (1 << (target + 1)) - 1
    reachable = [0] * (len(weights) + 1)
    reachable[-1] = 1
    for i in range(len(weights) - 1, -1, -1):
        suffix = reachable[i + 1]
        reachable[i] = suffix | ((suffix << weights[i]) & mask)
    dimensions["reachable_sums"] = reachable[0].bit_count()
    if not (reachable[0] >> target) & 1:
        return None, dimensions
    remaining, chosen = target, []
    for i, weight in enumerate(weights):
        if remaining == 0:
            break
        if weight <= remaining and (reachable[i + 1] >> (remaining - weight)) & 1:
            chosen.append(i)
            remaining -= weight
        else:
            base.require((reachable[i + 1] >> remaining) & 1, "broken exact-search witness")
    base.require(remaining == 0, "nonzero reconstructed residual")
    return chosen, dimensions


def solve_stratum(documents, quota, snapshot_limit=MAX_SNAPSHOT_BYTES):
    candidates = sorted((base.ranked(d) for d in documents), key=lambda d: (d["candidate_rank"], d["id"]))
    fit = [d for d in candidates if d["normalized_utf8_bytes"] <= quota]
    report = {"eligible_documents": len(candidates), "quota": quota,
              "rank_namespace": base.POLICY, "ranked_candidate_ids": [d["id"] for d in candidates]}
    if not fit:
        return None, {**report, "status": FAIL_NO_SOLUTION, "reason": "no_fitting_mandatory_coverage_document"}
    coverage = min(fit, key=lambda d: (d["normalized_utf8_bytes"], d["candidate_rank"], d["id"]))
    remaining = quota - coverage["normalized_utf8_bytes"]
    residual = [d for d in candidates if d["id"] != coverage["id"] and d["normalized_utf8_bytes"] <= remaining]
    chosen, dimensions = subset_indices([d["normalized_utf8_bytes"] for d in residual], remaining, snapshot_limit)
    report.update({"coverage_document_id": coverage["id"], "coverage_normalized_bytes": coverage["normalized_utf8_bytes"],
                   "excluded_exceeds_residual_ids": [d["id"] for d in candidates if d["id"] != coverage["id"]
                                                    and d["normalized_utf8_bytes"] > remaining],
                   "residual_candidate_ids": [d["id"] for d in residual], **dimensions})
    if chosen is None:
        return None, {**report, "status": FAIL_NO_SOLUTION, "reason": "exact_residual_unreachable"}
    selected_ids = {coverage["id"], *(residual[i]["id"] for i in chosen)}
    selected = [d for d in documents if d["id"] in selected_ids]
    report.update(status="EXACT_SOLUTION", selected_candidate_indices=chosen,
                  selected_ids_in_rank_order=[d["id"] for d in candidates if d["id"] in selected_ids])
    return selected, report


def solve_all(documents, allocation, snapshot_limit=MAX_SNAPSHOT_BYTES):
    base.require(len(documents) <= MAX_DOCUMENTS, "parent candidate bound exceeded")
    base.require(len({d["id"] for d in documents}) == len(documents), "duplicate ID")
    base.require(len({d["normalized_text_hash"] for d in documents}) == len(documents), "duplicate normalized text")
    groups = defaultdict(list)
    for d in documents:
        base.require(type(d["normalized_utf8_bytes"]) is int and d["normalized_utf8_bytes"] > 0, "invalid document length")
        groups[d["domain"], d["language"]].append(d)
    base.require(set(groups) == set(allocation), "parent strata mismatch")
    selected, reports = [], []
    for key, quota in sorted(allocation.items()):
        try:
            chosen, report = solve_stratum(groups[key], quota, snapshot_limit)
        except (ResourceLimit, MemoryError) as error:
            chosen, report = None, {"status": "SEARCH_INCOMPLETE_RESOURCE_LIMIT", "error": str(error)}
        reports.append({"domain": key[0], "language": key[1], **report})
        if chosen is not None:
            selected.extend(chosen)
        print(f"{key[0]}/{key[1]}: {report['status']}", flush=True)
    if any(r["status"] != "EXACT_SOLUTION" for r in reports):
        return None, reports
    # Every document fits in an exact selected subset. Reuse only the preserved
    # v1 rank/order/accounting rules, not its greedy selection on the parent pool.
    sample = base.construct(selected, allocation)
    for g in sample["strata"]:
        original = groups[g["domain"], g["language"]]
        g["eligible_documents"] = len(original)
        g["eligible_normalized_bytes"] = sum(d["normalized_utf8_bytes"] for d in original)
        g["skipped_documents"] = len(original) - g["selected_documents"]
    return sample, reports


def verify_witness(parent, allocation, sample):
    """Independent membership, integer accounting and order verifier; no solver calls."""
    originals = {d["id"]: d for d in parent}
    base.require(len(originals) == len(parent), "duplicate parent ID")
    docs = sample["ordered_documents"]
    base.require(len({d["id"] for d in docs}) == len(docs), "duplicate selected ID")
    base.require(len({d["normalized_text_hash"] for d in docs}) == len(docs), "duplicate selected text")
    groups, parent_groups = defaultdict(list), defaultdict(list)
    for d in parent:
        parent_groups[d["domain"], d["language"]].append(d)
    for index, d in enumerate(docs):
        original = originals.get(d["id"])
        base.require(original is not None, "selected document outside parent")
        base.require(set(d) == set(original) | {"candidate_rank", "coverage_document", "position"}, "unexpected witness fields")
        base.require(all(d[k] == v for k, v in original.items()), "altered document provenance/bytes")
        base.require(d["position"] == index and d["candidate_rank"] == base.ranked(original)["candidate_rank"], "rank/position mismatch")
        groups[d["domain"], d["language"]].append(d)
    base.require(set(groups) == set(allocation), "missing or extra stratum")
    base.require(len(sample["strata"]) == len(allocation), "missing stratum report")
    coverage = {}
    for key, quota in allocation.items():
        fit = [d for d in parent_groups[key] if d["normalized_utf8_bytes"] <= quota]
        base.require(fit, "missing mandatory coverage")
        expected = min(fit, key=lambda d: (d["normalized_utf8_bytes"], base.ranked(d)["candidate_rank"], d["id"]))
        coverage[key] = expected["id"]
        selected = groups[key]
        base.require(sum(d["normalized_utf8_bytes"] for d in selected) == quota, "per-stratum exact quota mismatch")
        base.require(expected["id"] in {d["id"] for d in selected}, "mandatory coverage missing")
        base.require(all(d["coverage_document"] == (d["id"] == expected["id"]) for d in selected), "coverage flag mismatch")
        g = next((g for g in sample["strata"] if (g["domain"], g["language"]) == key), None)
        base.require(g is not None and g["packing_status"] == g["exact_quota_status"] == "PASS", "stratum gate mismatch")
        required = {
            "allocated_normalized_bytes": quota, "actual_normalized_bytes": quota, "unused_quota_bytes": 0,
            "allocated_percent_of_cap": 100 * quota / sum(allocation.values()),
            "achieved_percent_of_actual": 100 * quota / sum(allocation.values()),
            "minimum_documents": 1, "minimum_normalized_bytes": (quota * 9 + 9) // 10,
            "selected_documents": len(selected), "eligible_documents": len(parent_groups[key]),
            "eligible_normalized_bytes": sum(d["normalized_utf8_bytes"] for d in parent_groups[key]),
            "skipped_documents": len(parent_groups[key]) - len(selected), "coverage_document_id": expected["id"],
            "source_utf8_bytes": sum(d["source_utf8_bytes"] for d in selected), "exhaustion_reason": "quota_filled",
            "ordered_document_ids": [d["id"] for d in selected], "global_positions": [d["position"] for d in selected],
            "last_global_position": selected[-1]["position"], "order_rule": "R", "early_exhaustion_allowed": True,
            "coverage_order_rank": base.digest({"policy": base.POLICY, "purpose": "coverage_order", "domain": key[0], "language": key[1]}),
            "ordered_list_sha256": base.digest([{k: d[k] for k in ("id", "normalized_text_hash", "normalized_utf8_bytes", "source_utf8_bytes")} for d in selected]),
        }
        base.require(all(g.get(k) == v for k, v in required.items()), "stratum accounting mismatch")
    # Independently compute the legal coverage ordering, without construct's queues.
    domains = sorted({k[0] for k in allocation})
    domain_keys = {domain: sorted((k for k in allocation if k[0] == domain), key=lambda k: (
        base.digest({"policy": base.POLICY, "purpose": "coverage_order", "domain": k[0], "language": k[1]}), k[1])) for domain in domains}
    coverage_ids = [coverage[domain_keys[domain][i]] for i in range(max(map(len, domain_keys.values())))
                    for domain in domains if i < len(domain_keys[domain])]
    base.require([d["id"] for d in docs[:len(allocation)]] == coverage_ids, "coverage order changed")
    served = {k: originals[coverage[k]]["normalized_utf8_bytes"] for k in allocation}
    queues = {k: sorted((d for d in groups[k] if d["id"] != coverage[k]), key=lambda d: (d["candidate_rank"], d["id"])) for k in allocation}
    for actual in docs[len(allocation):]:
        active = sorted(k for k in allocation if queues[k])
        base.require(active, "unexpected extra document")
        best = active[0]
        for key in active[1:]:
            if served[key] * allocation[best] < served[best] * allocation[key]:
                best = key
        expected = queues[best].pop(0)
        base.require(actual["id"] == expected["id"], "weighted-byte order changed")
        served[best] += expected["normalized_utf8_bytes"]
    base.require(not any(queues.values()), "missing ordered documents")
    base.require(sample["normalized_utf8_bytes"] == sum(allocation.values()) == sum(d["normalized_utf8_bytes"] for d in docs), "total exact byte mismatch")
    base.require(sample["source_utf8_bytes"] == sum(d["source_utf8_bytes"] for d in docs), "source-byte total mismatch")
    base.require(sample["selected_documents"] == len(docs) and sample["ordered_exposure_hash"] == base.digest(docs), "sample hash/count mismatch")
    return {"membership": "PASS", "whole_document_bytes": "PASS", "all_stratum_quotas": "PASS", "ordering": "PASS"}


def check_pins(plan):
    for path, expected in plan["input_hashes"].items():
        base.require(base.file_hash(path) == expected, f"pinned input changed: {path}")


def worker(attempt, mode):
    attempt = Path(attempt)
    plan = base.read_json(attempt / "plan.json")
    check_pins(plan)
    source, training, context = base.input_context(plan["selection"], plan["source"])
    documents, train_path = base.load_parent(plan["source"], source, training)
    sample, reports = solve_all(documents if mode == "solve" else list(reversed(documents)), base.quotas(), plan["snapshot_limit_bytes"])
    if mode == "solve":
        base.publish(attempt / "search.json", {"strata": reports, "all_exact": sample is not None})
        if sample is not None:
            base.publish(attempt / "candidate.json", {"status": "candidate_not_usable", "provenance": context,
                         "sample": sample, "policy_sha256": plan["policy_sha256"], "search": reports})
    else:
        candidate_hash = base.file_hash(attempt / "candidate.json")
        candidate = base.read_json(attempt / "candidate.json")
        base.require(candidate["provenance"] == context and candidate["policy_sha256"] == plan["policy_sha256"], "candidate provenance mismatch")
        verification = verify_witness(documents, base.quotas(), candidate["sample"])
        base.require(sample is not None and sample == candidate["sample"] and reports == candidate["search"],
                     "independent-process deterministic witness mismatch")
        base.require(base.file_hash(attempt / "candidate.json") == candidate_hash, "candidate changed during verification")
        base.publish(attempt / "verification.json", {"status": "PASS", "candidate_sha256": candidate_hash,
                     "checks": verification, "fresh_process_source_reload": True,
                     "include_first_reconstruction_repeated": True, "input_order_reversed_for_check": True})
    check_pins(plan)
    base.require(base.file_hash(train_path) == source["splits"]["train"]["sha256"], "training file changed")


def rejection(attempt, reason, detail=""):
    """A rejection has no final exposure SHA and never exposes a usable manifest."""
    rows = base.read_json(attempt / "search.json")["strata"] if (attempt / "search.json").exists() else []
    known = {(r["domain"], r["language"]): r for r in rows}
    all_rows = [known.get(key, {"domain": key[0], "language": key[1], "status": "NOT_COMPLETED"}) for key in sorted(base.quotas())]
    receipt = {"status": "exposure_rejected", "reason": reason, "detail": detail, "final_exposure_sha256": None,
               "plan_sha256": base.file_hash(attempt / "plan.json"), "strata": all_rows,
               "tokenizer_flop_preflight_run": False, "lm_training_run": False}
    base.publish(attempt / "receipt.json", receipt)
    return receipt


def run_child(attempt, mode, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ResourceLimit("total gate deadline reached")
    with (attempt / f"{mode}.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-m", "tools.phase_b_exact_exposure", "worker", "--attempt", str(attempt), "--mode", mode],
                                 cwd=base.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUTF8": "1"})
        try:
            code = child.wait(timeout=remaining)
        except BaseException:
            child.kill()
            child.wait()
            raise
    base.require(code == 0, f"{mode} worker failed with exit {code}; see {mode}.log")


def publish_success(attempt, plan):
    check_pins(plan)
    base.require(base.read_json(attempt / "plan.json") == plan, "execution plan changed")
    base.require(base.file_hash(attempt / "policy.json") == plan["policy_sha256"], "policy snapshot changed")
    candidate = base.read_json(attempt / "candidate.json")
    verification = base.read_json(attempt / "verification.json")
    base.require(verification["status"] == "PASS" and verification["candidate_sha256"] == base.file_hash(attempt / "candidate.json"), "missing or stale independent verification")
    base.require(base.read_json(attempt / "search.json")["all_exact"] is True, "incomplete search")
    base.require(len(candidate["search"]) == len(base.quotas()) == len(candidate["sample"]["strata"])
                 and {(r["domain"], r["language"]) for r in candidate["search"]} == set(base.quotas())
                 and all(r["status"] == "EXACT_SOLUTION" for r in candidate["search"]), "incomplete stratum solutions")
    body = {"exposure_schema_version": 2, "status": "exposure_frozen", "gate": "exact_packing_only",
            "policy_id": POLICY, "policy_sha256": plan["policy_sha256"], "plan_sha256": base.file_hash(attempt / "plan.json"),
            "provenance": candidate["provenance"], "sample": candidate["sample"],
            "generator": plan["generator"], "search": candidate["search"], "verification": verification,
            "tokenizer_outputs_used_for_sampling": False, "token_counts_used_for_sampling": False, "flops_used_for_sampling": False,
            "validation_or_test_text_used_for_sampling": False, "selection_metrics_used": False, "lm_results_used": False,
            "tokenizer_flop_preflight_run": False, "lm_training_run": False, "authorized_for_lm_execution": False}
    manifest = {**body, "content_sha256": base.digest(body)}
    staging = attempt / ".publication-incomplete"
    staging.mkdir()
    base.publish(staging / "exposure.json", manifest)
    manifest_hash = base.file_hash(staging / "exposure.json")
    sample = candidate["sample"]
    receipt = {"status": "exact_packing_feasible", "final_exposure_sha256": manifest_hash,
               "policy_sha256": plan["policy_sha256"], "plan_sha256": base.file_hash(attempt / "plan.json"),
               "selection_sha256": base.SELECTION_SHA, "source_manifest_sha256": base.SOURCE_SHA,
               "normalized_utf8_bytes": sample["normalized_utf8_bytes"], "source_utf8_bytes": sample["source_utf8_bytes"],
               "selected_documents": sample["selected_documents"], "strata_exact": len(sample["strata"]),
               "independent_verification": "PASS", "tokenizer_flop_preflight_run": False, "lm_training_run": False}
    base.publish(staging / "receipt.json", receipt)
    base.publish(staging / "strata-report.json", {"manifest_sha256": manifest_hash, "strata": sample["strata"]})
    base.require(base.read_json(staging / "exposure.json") == manifest, "publication readback mismatch")
    check_pins(plan)
    # Atomic directory rename is the sole success commit point. Neither a
    # candidate nor an incomplete staging directory is an accepted exposure bundle.
    base.require(not (attempt / "frozen").exists(), "duplicate publication")
    os.rename(staging, attempt / "frozen")
    return receipt


def supervise(attempt, plan, child_runner=run_child):
    deadline = time.monotonic() + plan["timeout_seconds"]
    try:
        child_runner(attempt, "solve", deadline)
        base.require(base.read_json(attempt / "plan.json") == plan, "execution plan changed during search")
        search = base.read_json(attempt / "search.json")
        if not search["all_exact"]:
            reason = "NO_EXACT_SOLUTION" if all(r["status"] in ("EXACT_SOLUTION", FAIL_NO_SOLUTION) for r in search["strata"]) else "SEARCH_INCOMPLETE_RESOURCE_LIMIT"
            return rejection(attempt, reason)
        child_runner(attempt, "verify", deadline)
        base.require(base.read_json(attempt / "plan.json") == plan, "execution plan changed during verification")
        if time.monotonic() >= deadline:
            raise ResourceLimit("verification exceeded total gate deadline")
        return publish_success(attempt, plan)
    except (subprocess.TimeoutExpired, ResourceLimit, MemoryError) as error:
        return rejection(attempt, "SEARCH_INCOMPLETE_RESOURCE_LIMIT", str(error))
    except (KeyboardInterrupt, SystemExit):
        return rejection(attempt, "SEARCH_INTERRUPTED")
    except Exception as error:
        # Outer failure-evidence boundary only: never substitute, retry or continue.
        return rejection(attempt, "SEARCH_ERROR", f"{type(error).__name__}: {error}")


def generate(args):
    attempt = Path(args.output).resolve()
    base.require(not attempt.exists(), "duplicate attempt directory")
    attempt.mkdir(parents=True)
    base.publish(attempt / "initial-rejection-receipt.json", {"status": "exposure_rejected", "reason": "SEARCH_NOT_COMPLETED",
                 "final_exposure_sha256": None, "rule": "Unless a fully verified frozen/ bundle exists, this attempt is rejected. Hard termination or host shutdown cannot imply success."})
    try:
        base.require(base.file_hash(args.policy) == REVIEW_SHA, "unapproved exact-packing review")
        source, _, _ = base.input_context(args.selection, args.source)
        train_path = (Path(args.source).resolve().parent / source["splits"]["train"]["path"]).resolve()
        pinned = {str(Path(p).resolve()): base.file_hash(p) for p in (args.selection, args.source, args.policy, __file__, base.__file__)}
        pinned[str(train_path)] = source["splits"]["train"]["sha256"]
        policy = {"policy_id": POLICY, "review_sha256": REVIEW_SHA, "rank_namespace": base.POLICY,
                  "required_normalized_bytes": base.EXACT_BYTES, "quotas": [{"domain": k[0], "language": k[1], "bytes": v} for k, v in sorted(base.quotas().items())],
                  "coverage_rule": "shortest_fitting_then_rank_then_ID_unchanged_v1", "solver": "suffix_bitset_zero_one_include_first",
                  "ordering_rule": "R_unchanged_v1", "timeout_seconds": MAX_SECONDS, "snapshot_limit_bytes": MAX_SNAPSHOT_BYTES,
                  "limit_scope": "estimated_solver_snapshots_including_integer_overhead; not total process RSS",
                  "publication": "all_30_exact_plus_fresh_process_verification_atomic_frozen_bundle"}
        base.publish(attempt / "policy.json", policy)
        plan = {"schema_version": 2, "selection": str(Path(args.selection).resolve()), "source": str(Path(args.source).resolve()),
                "input_hashes": pinned, "policy_sha256": base.file_hash(attempt / "policy.json"),
                "timeout_seconds": MAX_SECONDS, "snapshot_limit_bytes": MAX_SNAPSHOT_BYTES,
                "generator": {"base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.ROOT, text=True).strip(),
                              "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=base.ROOT, text=True).strip()),
                              "source_hashes": {str(Path(p).relative_to(base.ROOT)): base.file_hash(p) for p in (Path(__file__).resolve(), Path(base.__file__).resolve())},
                              "python": sys.version, "unicode_database_version": base.unicodedata.unidata_version,
                              "frozen_lm_implementation_pin_bypassed": False}}
        base.publish(attempt / "plan.json", plan)
    except BaseException as error:
        base.publish(attempt / "setup-rejection-receipt.json", {"status": "exposure_rejected", "reason": "SETUP_NOT_COMPLETED",
                     "detail": f"{type(error).__name__}: {error}", "final_exposure_sha256": None})
        raise
    return supervise(attempt, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("generate")
    for name in ("selection", "source", "policy", "output"):
        create.add_argument(f"--{name}", type=Path, required=True)
    child = commands.add_parser("worker")
    child.add_argument("--attempt", type=Path, required=True)
    child.add_argument("--mode", choices=("solve", "verify"), required=True)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.attempt, args.mode)
    else:
        # Catchable termination seals rejection; hard termination leaves initial rejection.
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        result = generate(args)
        print(json.dumps({k: v for k, v in result.items() if k != "strata"}, indent=2))
        if result["status"] != "exact_packing_feasible":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
