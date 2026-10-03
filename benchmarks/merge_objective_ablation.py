"""Offline, fixed-pool SuperBPE objective component ablations (#87)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import math
from pathlib import Path
import unicodedata

from benchmarks import byte_fallback_analysis as source
from benchmarks import run_research_experiments as h
from benchmarks.analyze_tokenizer_failures import csv_text
from benchmarks.tokenizer_failure_metrics import analyze_condition
from uniqtoken.byte_codec import ByteFallbackEngine as Bytes
from uniqtoken.cem_merger import SuperBPE
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramModel

COMPONENTS = (
    "cross_entropy",
    "compression_gain",
    "document_frequency",
    "boundary_cost",
    "fallback_cost",
    "fragmentation_penalty",
)
MATRIX = {
    "pool_ce": {"cross_entropy": 1},
    "compression": {"compression_gain": -1},
    "frequency": {"document_frequency": -1},
    "boundary": {"boundary_cost": 1},
    "fallback": {"fallback_cost": 1},
    "fragmentation": {"fragmentation_penalty": 1},
    "compression_boundary_fragmentation": {"compression_gain": -1, "boundary_cost": 1, "fragmentation_penalty": 1},
    "ce_compression": {"cross_entropy": 1, "compression_gain": -1},
}


def boundary_kind(char, space):
    if char == space or char.isspace():
        return "whitespace"
    if unicodedata.category(char).startswith("P"):
        return "punctuation"
    return "other"


def candidate_pool(model, chunks, documents, space="\u2581"):
    """Use the current SuperBPE initial stream boundaries and eligibility."""
    encoded = {chunk: model.encode(chunk) for chunk in sorted(set(chunks))}
    streams, current = [], []
    for chunk in chunks:
        current.extend(encoded[chunk])
        if len(current) >= 200:
            streams.append(current)
            current = []
    if current:
        streams.append(current)
    counts = Counter(pair for stream in streams for pair in zip(stream, stream[1:]))
    total = sum(counts.values())
    doc_counts = Counter()
    for document in documents:
        tokens = [t for chunk in document for t in encoded[chunk]]
        doc_counts.update(set(zip(tokens, tokens[1:])))
    result = []
    for (a, b), frequency in sorted(counts.items()):
        merged = a + b
        if frequency < 2 or any(t in model.special_tokens or Bytes.is_byte_token(t) for t in (a, b)):
            continue
        if merged in model.vocab or len(merged) > model.max_subword_len:
            continue
        if space not in merged[1:] or not merged.strip(space):
            continue
        lp = math.log(frequency / total)
        ce = frequency * (model.vocab[a] + model.vocab[b] - lp)
        if ce >= 0:
            continue
        # Repeated equal tokens can overlap; count only disjoint replacements.
        gain = frequency
        if a == b:
            gain = 0
            for stream in streams:
                index = 0
                while index + 1 < len(stream):
                    hit = stream[index] == a and stream[index + 1] == b
                    gain += int(hit)
                    index += 2 if hit else 1
        left, right = boundary_kind(a[-1], space), boundary_kind(b[0], space)
        result.append(
            {
                "left": a,
                "right": b,
                "merged": merged,
                "frequency": frequency,
                "log_probability": lp,
                "cross_entropy": ce,
                "compression_gain": gain,
                "document_frequency": doc_counts[(a, b)],
                "boundary_cost": gain * int(left != right and "whitespace" in (left, right)),
                "fallback_cost": 0,
                "fragmentation_penalty": -gain * int(left == right and left in ("whitespace", "punctuation")),
            }
        )
    return result


def select_candidates(pool, condition, limit):
    h.require(condition in MATRIX and type(limit) is int and limit > 0, "invalid ablation selection")
    h.require(len(pool) >= limit, "candidate pool cannot meet exact merge reserve")
    scales = {name: max(abs(row[name]) for row in pool) or 1 for name in COMPONENTS}
    weights = MATRIX[condition]

    def score(row):
        return math.fsum(weight * row[name] / scales[name] for name, weight in weights.items())

    # One concatenation can have multiple decompositions; each consumes one slot.
    selected, seen = [], set()
    for row in sorted(pool, key=lambda r: (score(r), -r["frequency"], r["left"], r["right"])):
        if row["merged"] not in seen:
            selected.append({**row, "selection_score": score(row)})
            seen.add(row["merged"])
        if len(selected) == limit:
            return selected
    raise ValueError("unique candidate tokens cannot meet exact merge reserve")


def admit(model, selected):
    probs = {t: max(math.exp(p), 1e-300) for t, p in model.vocab.items()}
    ids = dict(model.token_to_id)
    for row in selected:
        token = row["merged"]
        h.require(token not in probs, "duplicate admitted token")
        probs[token] = math.exp(row["log_probability"])
        ids[token] = len(ids)
    total = math.fsum(probs.values())
    return UnigramModel(
        {t: math.log(p / total) for t, p in probs.items()},
        ids,
        {i: t for t, i in ids.items()},
        list(model.special_tokens),
        model.max_subword_len,
        model.byte_fallback,
        model.unk_token,
    )


def run(dataset, output, budget=8192, reserve=64):
    h.require(not output.exists(), "output must be new")
    h.require(not output.resolve().is_relative_to(dataset.resolve().parent), "output overlaps source")
    h.require(
        type(budget) is int and type(reserve) is int and budget > reserve + 260 and reserve > 0,
        "invalid budget/reserve",
    )
    identity = h.runtime_identity()
    h.require(not identity["working_tree_dirty"], "commit experiment before recording evidence")
    assignments, provenance = source.assignments(dataset)
    training = [t for _, t in assignments["train"]]
    print("Training shared seed", flush=True)
    base = source.train_base(training, budget - reserve)
    docs = [[*base.pre_tokenizer.pre_tokenize(base.normalizer.normalize(t)), h.SPECIALS[3]] for t in training]
    chunks = [c for document in docs for c in document]
    pool = candidate_pool(base.model, chunks, docs)
    pool_hash = h.digest(pool)
    output.mkdir(parents=True)
    all_records, conditions = [], {}
    for condition in ("current_superbpe", *MATRIX):
        print(f"Evaluating {condition}", flush=True)
        if condition == "current_superbpe":
            optimizer = SuperBPE(max_merges=reserve)
            model = optimizer.optimize(base.model, chunks)
            selected = [asdict(r) for r in optimizer.merge_provenance]
        else:
            selected = select_candidates(pool, condition, reserve)
            model = admit(base.model, selected)
        tok = CustomTokenizer(base.normalizer, base.pre_tokenizer, model)
        adapter = h.ResearchTokenizer("uniq_superbpe", tok, model.token_to_id, len(selected))
        h.validate_tokenizer(adapter, budget)
        h.require(all(model.token_to_id[t] == i for t, i in base.model.token_to_id.items()), "base IDs changed")
        directory = output / condition
        tok.save(directory, save_binary=False)
        records = analyze_condition(adapter, assignments)
        all_records.extend({**row, "condition": condition} for row in records)
        conditions[condition] = {
            "selected": selected,
            "actual_vocab_size": len(model.vocab),
            "model_hashes": h.artifact_hashes(directory),
        }
    h.require(h.runtime_identity() == identity, "source/runtime changed during experiment")
    payload = {
        "schema_version": 1,
        "result_label": "EXPLANATORY_ABLATION",
        "identity": identity,
        "assignments": provenance,
        "configuration": {"budget": budget, "reserve": reserve, "matrix": MATRIX, "randomness": "none", "seed": None},
        "pool_sha256": pool_hash,
        "pool": pool,
        "conditions": conditions,
        "records": all_records,
        "constant_components": [name for name in COMPONENTS if len({row[name] for row in pool}) <= 1],
    }
    h.write_new_json(output / "results.json", payload)
    (output / "metrics.csv").write_text(csv_text(all_records), encoding="utf-8")
    (output / "REPORT.md").write_text(report(payload), encoding="utf-8")
    h.write_new_json(output / "manifest.json", {"status": "complete", "artifacts": h.artifact_hashes(output)})
    return payload


def report(payload):
    metrics = (
        "bytes_per_token",
        "byte_fallback_percent",
        "tokens_per_unicode_character",
        "whitespace_split_run_percent",
        "punctuation_split_run_percent",
    )
    rows = [r for r in payload["records"] if r["split"] == "validation"]
    lines = [
        "# SuperBPE component ablations",
        "",
        "Explanatory tokenizer-only measurements. No final objective, downstream claim or held-out test access.",
        "All fixed-pool ablations see identical candidates. Current SuperBPE is the dynamic reference; pool_ce controls for freezing its candidate pool and scoring state.",
        "Constant components cannot be identified by this pool: " + ", ".join(payload["constant_components"]),
        "",
        "| Condition | BpT | Fallback % | Tokens/character | Whitespace split % | Punctuation split % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["scope"] == "aggregate":
            lines.append(
                "| "
                + row["condition"]
                + " | "
                + " | ".join("NA" if row[k] is None else f"{row[k]:.5f}" for k in metrics)
                + " |"
            )
    reference = {
        (r["domain"], r["language"]): r
        for r in rows
        if r["scope"] == "stratum" and r["condition"] == "current_superbpe"
    }
    lines.extend(
        [
            "",
            "## Per-stratum trade-offs",
            "",
            "Counts compare observed validation strata with current SuperBPE. A BpT loss above 1% is reported as a regression, not hidden by the aggregate.",
            "",
            "| Condition | Strata with >1% BpT loss | Strata with more fallback | Strata with more punctuation splitting |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for condition in MATRIX:
        selected = [r for r in rows if r["scope"] == "stratum" and r["condition"] == condition]
        loss = fallback = fragmentation = 0
        for row in selected:
            base = reference[(row["domain"], row["language"])]
            loss += row["bytes_per_token"] < base["bytes_per_token"] * 0.99
            fallback += row["byte_fallback_percent"] > base["byte_fallback_percent"]
            a, b = row["punctuation_split_run_percent"], base["punctuation_split_run_percent"]
            fragmentation += a is not None and b is not None and a > b
        lines.append(f"| {condition} | {loss} | {fallback} | {fragmentation} |")
    lines.extend(
        [
            "",
            "The CSV/JSON retain every language/domain/stratum, exact histograms and normalization counts. Missing validation domains are not inferred from training observations.",
            "Component values are training proxies. Re-encoding and greedy interactions can differ from their first-order predictions. No aggregate result selects a production objective.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=8192)
    parser.add_argument("--reserve", type=int, default=64)
    args = parser.parse_args()
    run(args.dataset, args.output, args.budget, args.reserve)


if __name__ == "__main__":
    main()
