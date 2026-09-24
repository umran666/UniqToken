"""Independently check a completed preflight using saved per-document counts.

No encoders, samplers, training, or evaluation are invoked. Original evidence is
read-only; the verification is a new provenance-linked JSON artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks import run_research_experiments as h


def reference_schedule(counts, vocab, regime, budget, context=128):
    """Scalar dense-matmul accounting, independent of the harness scheduler."""
    h.require(counts and all(type(n) is int and n > 0 for n in counts), "invalid saved token counts")
    h.require(regime in ("flops", "bytes") and budget > 0, "invalid verification budget")

    # Frozen B-SCREEN: two layers, d=128, feed-forward=512; untied output head.
    def components(n):
        core = 6 * 2 * n * (4 * 128**2 + 2 * 128 * 512) + 12 * 2 * n**2 * 128
        return core, 6 * n * 128 * vocab

    accounted_docs = steps = targets = squares = core = output = 0
    terminal_targets = 0
    while regime == "flops" or accounted_docs < len(counts):
        remaining = counts[accounted_docs % len(counts)] + 1
        terminal_targets = 0
        while remaining:
            n = min(context, remaining)
            if regime == "flops":
                while n and core + output + sum(components(n)) > budget:
                    n -= 1
                if not n:
                    h.require(core + output >= budget * 0.99, "unreachable budget")
                    break
            a, b = components(n)
            core += a
            output += b
            steps += 1
            targets += n
            squares += n * n
            remaining -= n
            terminal_targets += n
            if regime == "flops" and core + output >= budget * 0.99:
                break
        if regime == "flops" and core + output >= budget * 0.99:
            break
        accounted_docs += 1
    fully_predicted_docs = accounted_docs
    if regime == "flops" and terminal_targets == counts[accounted_docs % len(counts)] + 1:
        fully_predicted_docs += 1
    return {
        "training_steps": steps,
        "training_target_tokens": targets,
        "training_sequence_length_squared_sum": squares,
        "core_analytical_flops": core,
        "output_projection_flops": output,
        "actual_analytical_flops": core + output,
        "completed_training_documents": accounted_docs,
        "fully_predicted_documents": fully_predicted_docs,
        "terminal_document_target_tokens": terminal_targets if regime == "flops" else 0,
    }


def verify(report_path, expected_sha, snapshot, output_path):
    report_path, snapshot, output_path = map(Path, (report_path, snapshot, output_path))
    h.require(h.file_hash(report_path) == expected_sha, "preflight report hash mismatch")
    report = h.read_json(report_path)
    h.require(
        report["content_sha256"] == h.digest({k: v for k, v in report.items() if k != "content_sha256"}),
        "preflight content hash mismatch",
    )
    version = report["preflight_schema_version"]
    h.require(version in (2, 3), "unsupported preflight evidence schema")

    def check_inputs():
        h.require(h.file_hash(report_path) == expected_sha, "preflight report changed during verification")
        for path, expected in report["input_hashes"].items():
            original = Path(path)
            checked = snapshot if original.name == "phase_b_flop_preflight.py" else original
            h.require(h.file_hash(checked) == expected, f"changed preflight input: {path}")

    check_inputs()

    def pinned_path(sha):
        paths = [Path(path) for path, value in report["input_hashes"].items() if value == sha]
        h.require(len(paths) == 1, "ambiguous/missing input pin")
        return paths[0]

    exposure = h.read_json(pinned_path(report["exposure_sha256"]))["sample"]
    selection = h.read_json(pinned_path(report["selection_sha256"]))
    phase_a = pinned_path(report["phase_a_ledger_sha256"])
    h.require(len(report["conditions"]) == len(selection["conditions"]) == 9, "incomplete conditions")
    rows, counter_notes = [], []
    for index, (record, selected) in enumerate(zip(report["conditions"], selection["conditions"])):
        h.require(
            h.read_json(report_path.parent / f"condition-{index:03d}.json") == record, "condition evidence mismatch"
        )
        h.require(
            (record["tokenizer"], record["vocab_budget"]) == (selected["tokenizer"], selected["vocab_budget"]),
            "condition assignment mismatch",
        )
        h.require(
            h.artifact_hashes(phase_a.parent / selected["artifact"]) == selected["artifact_hashes"],
            "tokenizer artifact changed",
        )
        h.require(record["model_config"] == h.model_config("B", "cpu"), "model config changed")
        vocab = record["vocab_budget"]
        core_params = 2 * (4 * 128**2 + 2 * 128 * 512 + 512 + 9 * 128) + 128 * 128 + 2 * 128
        parameters = {
            "core_params": core_params,
            "non_embedding_params": core_params,
            "input_embedding_params": 128 * vocab,
            "output_head_params": 128 * vocab,
            "total_params": core_params + 256 * vocab,
        }
        h.require(all(record[key] == value for key, value in parameters.items()), "parameter accounting mismatch")
        h.require(
            record["tokenizer_config"] == selected["tokenizer_config"]
            and record["actual_vocab_size"] == selected["actual_vocab_size"] == vocab
            and record["special_tokens"] == selection["special_tokens"],
            "tokenizer config/budget mismatch",
        )
        token_rows = record["encoded_document_accounting"]
        h.require(len(token_rows) == len(exposure["ordered_documents"]), "incomplete document encoding")
        for token, document in zip(token_rows, exposure["ordered_documents"]):
            h.require(
                all(
                    token[field] == document[field]
                    for field in (
                        "position",
                        "id",
                        "normalized_text_hash",
                        "normalized_utf8_bytes",
                        "source_utf8_bytes",
                    )
                ),
                "encoded document/order mismatch",
            )
        counts = [d["text_tokens"] for d in token_rows]
        runs = {}
        for regime in ("flops", "bytes"):
            original = record["regimes"][regime]
            computed = reference_schedule(counts, record["vocab_budget"], regime, report["budgets"][regime])
            h.require(
                all(original[key] == value for key, value in computed.items() if key != "fully_predicted_documents"),
                "independent schedule/FLOP disagreement",
            )
            count = computed["completed_training_documents"]
            sizes = [{key: d[key] for key in (h.BYTE_BUDGET_FIELD, h.BYTE_AUDIT_FIELD)} for d in token_rows]
            h.require(
                original["complete_document_bytes"] == h.byte_totals(sizes, count), "byte accounting disagreement"
            )
            if regime == "bytes":
                h.require(
                    count == len(counts) and original["complete_document_bytes"][h.BYTE_BUDGET_FIELD] == 1_000_000,
                    "byte exposure mismatch",
                )
            coverage = min(30, computed["fully_predicted_documents"])
            minimum = 30 if version == 2 else 1
            if version == 3:
                h.require(
                    original["coverage_policy_version"] == "flop_fixed_budget_one_complete_document_v1"
                    and original["minimum_fully_predicted_documents"] == 1
                    and original["fully_predicted_documents"] == computed["fully_predicted_documents"],
                    "coverage policy/accounting mismatch",
                )
                h.require(
                    original["coverage_gate"] == ("PASS" if coverage >= minimum else "BLOCKED"),
                    "coverage gate mismatch",
                )
                if regime == "flops":
                    target = report["budgets"]["flops"]
                    h.require(
                        original["flop_target"] == target
                        and original["flop_tolerance"] == 0.01
                        and original["flop_interval"] == [0.99 * target, target],
                        "FLOP interval mismatch",
                    )
            runs[regime] = {
                **computed,
                "complete_document_bytes": original["complete_document_bytes"],
                "coverage_documents_fully_predicted": coverage,
                "coverage_gate": "PASS" if coverage >= minimum else "BLOCKED",
            }
            if coverage != original["coverage_documents_completed"]:
                counter_notes.append(
                    {
                        "tokenizer": record["tokenizer"],
                        "vocab_budget": record["vocab_budget"],
                        "regime": regime,
                        "byte_accounted_coverage": original["coverage_documents_completed"],
                        "fully_predicted_coverage": coverage,
                        "reason": "terminal document fully predicted; existing scheduler byte counter excludes it",
                    }
                )
        rows.append({"tokenizer": record["tokenizer"], "vocab_budget": record["vocab_budget"], "regimes": runs})
    blocked = [r for r in rows if r["regimes"]["flops"]["coverage_gate"] == "BLOCKED"]
    result = {
        "verification_schema_version": 1,
        "accounting_verification": "PASS",
        "status": "preflight_rejected"
        if blocked or report["runtime_blockers"] or report["upstream_source_truncations_for_review"]
        else "preflight_feasible",
        "original_report_sha256": expected_sha,
        "verifier_sha256": h.file_hash(__file__),
        "preflight_implementation_sha256": h.file_hash(snapshot),
        "source_manifest_sha256": report["source_manifest_sha256"],
        "exposure_sha256": report["exposure_sha256"],
        "selection_sha256": report["selection_sha256"],
        "preflight_identity": report["preflight_identity"],
        "runtime_blockers": report["runtime_blockers"],
        "upstream_source_truncations_for_review": report["upstream_source_truncations_for_review"],
        "counter_boundary_notes": counter_notes,
        "conditions": rows,
        "fully_predicted_coverage_blockers": [
            {"tokenizer": r["tokenizer"], "vocab_budget": r["vocab_budget"]} for r in blocked
        ],
        "original_evidence_unchanged": True,
        "tokenizers_reexecuted": False,
        "test_data_opened": False,
        "authorized_for_lm_execution": False,
    }
    check_inputs()
    h.write_new_json_atomic(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "report-sha256", "source-snapshot", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    result = verify(args.report, args.report_sha256, args.source_snapshot, args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
