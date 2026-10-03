"""Frozen tokenizer-only byte fallback experiment (#86).

Recovery is an offline candidate, never a runtime byte-token format change.
Complete Unicode scalars are admitted atomically; incomplete UTF-8 prefixes
cannot consume vocabulary slots. See BYTE_FALLBACK_ANALYSIS.md for the protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import math
from pathlib import Path
import time

from benchmarks import run_research_experiments as h
from benchmarks import run_phase_a as stages
from benchmarks.analyze_tokenizer_failures import guard_split_paths, csv_text
from uniqtoken.byte_codec import ByteFallbackEngine as Bytes
from uniqtoken.cem_merger import SuperBPE
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramModel

VERSION = "issue86-atomic-recovery-v2"
CONDITIONS = ("baseline", "atomic_recovery", "fallback_weighted")
DEFAULT_BUDGETS = (8192, 16384, 32768)
MAX_REGRESSION_PCT = 1.0


def extract_fallback_spans(tokens):
    """Lengths in bytes, with canonical one-byte fallback leaves."""
    spans, run = [], 0
    for token in tokens:
        if Bytes.is_byte_token(token):
            run += 1
        else:
            if run:
                spans.append(run)
            run = 0
    if run:
        spans.append(run)
    return spans


def compute_span_metrics(spans):
    counts = Counter(spans)
    n = len(spans)
    ordered = sorted(spans)
    return {
        "count": n,
        "mean": sum(spans) / n if n else None,
        "p50": ordered[math.ceil(n * 0.5) - 1] if n else None,
        "p95": ordered[math.ceil(n * 0.95) - 1] if n else None,
        "max": max(spans) if n else None,
        "histogram_bytes": {str(k): v for k, v in sorted(counts.items())},
    }


def evaluate_regressions(baseline, candidate, strata, threshold=MAX_REGRESSION_PCT):
    h.require(math.isfinite(threshold) and threshold >= 0, "invalid regression threshold")
    h.require(bool(strata), "no reference strata")
    regressions = {}
    for key in sorted(strata):
        h.require(key in baseline and key in candidate, f"missing reference stratum: {key}")
        a, b = baseline[key]["bytes_per_token"], candidate[key]["bytes_per_token"]
        h.require(a is not None and b is not None and a > 0 and b > 0, "undefined compression")
        h.require(math.isfinite(a) and math.isfinite(b), "nonfinite compression")
        regressions[key] = max(0.0, 100 * (a - b) / a)
    return all(x <= threshold for x in regressions.values()), regressions


def recovery_candidates(model, chunks, fallback_weight=0.0):
    """Rank whole missing scalars using train-only fallback occurrence counts.

    score = f * (sum byte log P - log(f/N)) - weight * f * UTF8_length.
    N counts ALL training token emissions. Lower scores are selected first.
    The added term has units weight*n_fallback_bytes, not a probability.
    This is a fixed-candidate admission stage before ordinary SuperBPE.
    """
    h.require(math.isfinite(fallback_weight) and fallback_weight >= 0, "invalid fallback weight")
    counts: Counter[str] = Counter()
    total = 0
    for chunk in chunks:
        tokens = model.encode(chunk)
        total += len(tokens)
        pending = bytearray()
        for token in [*tokens, None]:
            if token is not None and Bytes.is_byte_token(token):
                pending.append(Bytes.token_to_byte(token))
                continue
            if pending:
                # Model fallback leaves must cover complete source characters.
                counts.update(pending.decode("utf-8", errors="strict"))
                pending.clear()
    rows = []
    for char, frequency in sorted(counts.items()):
        raw = char.encode("utf-8")
        if frequency < 2 or char in model.vocab or char in model.special_tokens or len(raw) < 2:
            continue
        byte_tokens = [Bytes.byte_to_token(b) for b in raw]
        log_probability = math.log(frequency / total)
        ce = frequency * (math.fsum(model.vocab[b] for b in byte_tokens) - log_probability)
        rows.append(
            {
                "token": char,
                "frequency": frequency,
                "utf8_bytes": len(raw),
                "score": ce - fallback_weight * frequency * len(raw),
                "cross_entropy_term": ce,
                "log_probability": log_probability,
                "byte_tokens": byte_tokens,
            }
        )
    return sorted(rows, key=lambda row: (row["score"], -row["frequency"], row["token"]))


def recover_characters(model, chunks, limit, fallback_weight=0.0):
    h.require(type(limit) is int and limit >= 0, "invalid recovery limit")
    selected = recovery_candidates(model, chunks, fallback_weight)[:limit]
    if not selected:
        return model, []
    probs = {token: max(math.exp(lp), 1e-300) for token, lp in model.vocab.items()}
    for row in selected:
        probs[row["token"]] = math.exp(row["log_probability"])
    total = math.fsum(probs.values())
    ids = dict(model.token_to_id)
    for row in selected:
        ids[row["token"]] = len(ids)
    updated = UnigramModel(
        vocab={token: math.log(p / total) for token, p in probs.items()},
        token_to_id=ids,
        id_to_token={i: token for token, i in ids.items()},
        special_tokens=list(model.special_tokens),
        max_subword_len=model.max_subword_len,
        byte_fallback=model.byte_fallback,
        unk_token=model.unk_token,
    )
    return updated, selected


def evaluate_stratum(tok, texts):
    total_tokens, fallback, total_bytes = 0, 0, 0
    spans = []
    emissions: Counter[str] = Counter()
    for text in texts:
        tokens = tok.encode(text)
        h.require(tok.decode_tokens(tokens) == text, "normalized roundtrip failure")
        h.require(all(token in tok.model.token_to_id for token in tokens), "unknown emission")
        total_tokens += len(tokens)
        total_bytes += len(text.encode("utf-8"))
        fallback += sum(Bytes.is_byte_token(t) for t in tokens)
        spans.extend(extract_fallback_spans(tokens))
        emissions.update(tokens)
    return {
        "documents": len(texts),
        "total_tokens": total_tokens,
        "normalized_utf8_bytes": total_bytes,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else None,
        "fallback_tokens": fallback,
        "fallback_pct": 100 * fallback / total_tokens if total_tokens else None,
        "span_stats": compute_span_metrics(spans),
    }, emissions


def select_records(rows, texts, count, characters):
    h.require(count > 0 and characters > 0, "positive sample limits required")
    seen: Counter[str] = Counter()
    chosen = []
    for row, text in zip(rows, texts):
        key = row["domain"] + ":" + row["language"]
        if seen[key] < count:
            selected = text[:characters]
            if selected:
                chosen.append((row, selected))
                seen[key] += 1
    return chosen


def assignments(dataset, count=32, characters=2048):
    guard_split_paths(dataset, h.read_json(dataset))
    tr, train, vr, val, source = stages.load_stage_source(dataset)
    pairs = {
        "train": select_records(tr, train, count, characters),
        "validation": select_records(vr, val, count, characters),
    }
    h.require(not {r["id"] for r, _ in pairs["train"]} & {r["id"] for r, _ in pairs["validation"]}, "ID overlap")
    h.require(
        not {h.digest(t) for _, t in pairs["train"]} & {h.digest(t) for _, t in pairs["validation"]}, "excerpt overlap"
    )
    provenance = {
        "source": source,
        "selection": {"first_documents_per_stratum": count, "normalized_prefix_characters": characters},
        "splits": {
            split: [
                {
                    "id": r["id"],
                    "language": r["language"],
                    "domain": r["domain"],
                    "normalized_excerpt_sha256": h.digest(t),
                    "normalized_utf8_bytes": len(t.encode("utf-8")),
                }
                for r, t in records
            ]
            for split, records in pairs.items()
        },
    }
    return pairs, provenance


def train_base(texts, budget):
    tok = CustomTokenizer.train_from_corpus(
        texts,
        target_vocab_size=budget,
        min_frequency=1,
        special_tokens=list(h.SPECIALS),
        min_edge_log_prob=float("-inf"),
        verbose=False,
    )
    order = [*h.SPECIALS, *(t for t in tok.model.token_to_id if t not in h.SPECIAL_IDS)]
    tok.model.token_to_id = {t: i for i, t in enumerate(order)}
    tok.model.id_to_token = {i: t for i, t in enumerate(order)}
    h.validate_tokenizer(h.ResearchTokenizer("uniq_unigram", tok, tok.model.token_to_id), budget)
    return tok


def make_condition(base, chunks, budget, reserve, condition, recovery_limit, weight):
    h.require(condition in CONDITIONS, "unknown condition")
    model, recovered = base.model, []
    if condition != "baseline":
        model, recovered = recover_characters(
            model, chunks, recovery_limit, weight if condition == "fallback_weighted" else 0
        )
    optimizer = SuperBPE(max_merges=reserve - len(recovered))
    model = optimizer.optimize(model, chunks)
    tok = CustomTokenizer(normalizer=base.normalizer, pre_tokenizer=base.pre_tokenizer, model=model)
    h.validate_tokenizer(h.ResearchTokenizer("uniq_superbpe", tok, model.token_to_id, len(optimizer.merges)), budget)
    h.require(all(model.token_to_id[t] == i for t, i in base.model.token_to_id.items()), "existing ID drift")
    return tok, recovered, [asdict(row) for row in optimizer.merge_provenance]


def run_benchmark(
    dataset, output, budgets=DEFAULT_BUDGETS, count=32, characters=2048, reserve=64, recovery_limit=16, weight=5.0
):
    h.require(not output.exists(), "output must be new")
    h.require(not output.resolve().is_relative_to(dataset.resolve().parent), "output overlaps frozen source")
    h.require(
        len(set(budgets)) == len(budgets) and all(type(b) is int and b > reserve + 260 for b in budgets),
        "invalid budgets",
    )
    h.require(0 <= recovery_limit < reserve, "recovery must leave ordinary merge capacity")
    h.require(math.isfinite(weight) and weight >= 0, "invalid fallback weight")
    identity = h.runtime_identity()
    h.require(not identity["working_tree_dirty"], "commit source before recording evidence")
    pairs, provenance = assignments(dataset, count, characters)
    train = [t for _, t in pairs["train"]]
    val = defaultdict(list)
    for row, text in pairs["validation"]:
        val[row["domain"] + ":" + row["language"]].append(text)
    output.mkdir(parents=True)
    result = {
        "version": VERSION,
        "identity": identity,
        "assignments": provenance,
        "configuration": {
            "budgets": list(budgets),
            "reserve": reserve,
            "recovery_limit": recovery_limit,
            "fallback_weight": weight,
            "max_regression_pct": MAX_REGRESSION_PCT,
            "seed": None,
            "randomness": "none",
        },
        "validation_absent_training_strata": sorted(
            {r["domain"] + ":" + r["language"] for r, _ in pairs["train"]} - set(val)
        ),
        "runs": {},
    }
    flat = []
    for budget in budgets:
        print(f"Training shared seed for {budget}", flush=True)
        started = time.perf_counter()
        base = train_base(train, budget - reserve)
        base_seconds = time.perf_counter() - started
        chunks = [
            c
            for text in train
            for c in [*base.pre_tokenizer.pre_tokenize(base.normalizer.normalize(text)), h.SPECIALS[3]]
        ]
        baseline = None
        rows = {}
        for condition in CONDITIONS:
            print(f"Evaluating {budget}: {condition}", flush=True)
            started = time.perf_counter()
            tok, recovered, merges = make_condition(base, chunks, budget, reserve, condition, recovery_limit, weight)
            seconds = time.perf_counter() - started
            directory = output / f"{budget}-{condition}"
            tok.save(directory, save_binary=False)
            metrics = {}
            observed: Counter[str] = Counter()
            for stratum, texts in sorted(val.items()):
                metrics[stratum], counts = evaluate_stratum(tok, texts)
                observed.update(counts)
                sm = metrics[stratum]
                flat.append(
                    {
                        "budget": budget,
                        "condition": condition,
                        "stratum": stratum,
                        **{k: v for k, v in sm.items() if k != "span_stats"},
                        **{f"span_{k}": sm["span_stats"][k] for k in ("count", "mean", "p50", "p95", "max")},
                    }
                )
            if baseline is None:
                baseline = metrics
            passed, regressions = evaluate_regressions(baseline, metrics, set(val))
            additions = [r["token"] for r in recovered] + [r["merged"] for r in merges]
            training_counts = Counter(t for text in train for t in tok.encode(text))
            rows[condition] = {
                "actual_vocab_size": len(tok.model.vocab),
                "shared_base_training_seconds": base_seconds,
                "optimization_seconds": seconds,
                "artifact_hashes": h.artifact_hashes(directory),
                "recovered": recovered,
                "merges": merges,
                "strata": metrics,
                "unobserved_validation_additions": sum(observed[t] == 0 for t in additions),
                "unobserved_training_additions": sum(training_counts[t] == 0 for t in additions),
                "incomplete_prefix_additions": sum(t.startswith("<0x") for t in additions),
                "regression_gate_passed": passed,
                "regressions_pct": regressions,
                "emission_counts": {t: {"train": training_counts[t], "validation": observed[t]} for t in additions},
            }
        result["runs"][str(budget)] = rows
    h.require(h.runtime_identity() == identity, "source/runtime changed during experiment")
    h.write_new_json(output / "results.json", result)
    (output / "fallback_metrics.csv").write_text(csv_text(flat), encoding="utf-8")
    (output / "REPORT.md").write_text(report(result), encoding="utf-8")
    h.write_new_json(output / "manifest.json", {"status": "complete", "artifacts": h.artifact_hashes(output)})
    return result


def report(result):
    lines = [
        "# Atomic fallback recovery diagnostic",
        "",
        "Tokenizer-only descriptive experiment; no held-out test access or LM result.",
        "All conditions share seed vocabulary, frozen training excerpts, validation excerpts, normalization and exact final budget.",
        "Recovery admits whole missing Unicode scalars before the remaining ordinary SuperBPE merges. It is a separate admission policy, not pairwise byte-prefix merging.",
        "The regression gate checks every observed validation stratum at the predeclared 1% BpT loss threshold.",
        "Missing validation strata cannot be certified. Unobserved validation additions are not proof of intrinsically dead tokens.",
        "",
    ]
    for budget, rows in result["runs"].items():
        lines.extend(
            [
                f"## Vocabulary budget {budget}",
                "",
                "| Condition | BpT | Fallback % | Recovery slots | Training-unobserved | Validation-unobserved | Worst BpT regression % | Gate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for name, row in rows.items():
            strata = list(row["strata"].values())
            tokens = sum(s["total_tokens"] for s in strata)
            bpt = sum(s["normalized_utf8_bytes"] for s in strata) / tokens
            fallback = 100 * sum(s["fallback_tokens"] for s in strata) / tokens
            lines.append(
                f"| {name} | {bpt:.5f} | {fallback:.5f} | {len(row['recovered'])} | {row['unobserved_training_additions']} | {row['unobserved_validation_additions']} | {max(row['regressions_pct'].values()):.4f} | {row['regression_gate_passed']} |"
            )
        baseline = rows["baseline"]["strata"]
        for name in CONDITIONS[1:]:
            strata = rows[name]["strata"]
            fewer = sum(strata[s]["fallback_tokens"] < baseline[s]["fallback_tokens"] for s in baseline)
            longer = sum(
                strata[s]["span_stats"]["p95"] is not None
                and baseline[s]["span_stats"]["p95"] is not None
                and strata[s]["span_stats"]["p95"] > baseline[s]["span_stats"]["p95"]
                for s in baseline
            )
            lines.extend(
                [
                    "",
                    f"{name}: {fewer}/{len(baseline)} strata emit fewer fallback bytes; {longer} have a longer fallback-span p95.",
                ]
            )
        unweighted, weighted = rows["atomic_recovery"], rows["fallback_weighted"]
        equal = unweighted["strata"] == weighted["strata"]
        lines.extend(
            [
                "",
                f"{budget}: weighted and unweighted validation metrics are {'identical (no demonstrated weighting benefit)' if equal else 'different; inspect all strata and gates, without selecting a winner on aggregate alone'}.",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "Per-stratum fallback frequency and contiguous byte-span p50/p95/max are in fallback_metrics.csv; exact histograms, merge records and token utilization audits are in results.json.",
            "The following training domain/language pairs are absent from validation, which uses distinct FLORES domain labels: "
            + ", ".join(result["validation_absent_training_strata"]),
            "The original PR's approximate-budget numbers and hard-coded positive conclusions are withdrawn. These data do not establish a general multilingual advantage or a downstream-quality improvement.",
            "",
        ]
    )
    splits = result["assignments"]["splits"]
    missing_languages = sorted({r["language"] for r in splits["train"]} - {r["language"] for r in splits["validation"]})
    lines.extend(["Languages without any validation coverage: " + ", ".join(missing_languages) + ".", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--documents-per-stratum", type=int, default=32)
    parser.add_argument("--characters", type=int, default=2048)
    parser.add_argument("--reserve", type=int, default=64)
    parser.add_argument("--recovery-limit", type=int, default=16)
    args = parser.parse_args()
    run_benchmark(
        args.dataset,
        args.output,
        args.budgets,
        args.documents_per_stratum,
        args.characters,
        args.reserve,
        args.recovery_limit,
    )


if __name__ == "__main__":
    main()
