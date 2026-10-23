"""Controlled benchmark comparison between ReferenceMergeEngine, FastMergeEngine, and Production (#115).

Measures runtime, throughput, and verifies zero-mismatch differential parity
across diverse sequence lengths and merge densities with warm-up and statistical spread.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from uniqtoken.merge_engine import (
    FastMergeEngine,
    MembershipMergeTable,
    MergeEngine,
    MergeConstraints,
    ReferenceMergeEngine,
    SemanticProfile,
    apply_engine_to_pieces,
    cross_word_membership_table,
)
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramModel

S = "\u2581"


def _create_baseline_benchmark_tokenizer(model: UnigramModel) -> CustomTokenizer:
    """Instantiate a production baseline tokenizer with merge engine disabled."""
    return CustomTokenizer(
        Normalizer(normalize_unicode=False),
        RegexPreTokenizer(),
        model,
        merge_engine="default",
    )


def build_synthetic_benchmark_fixture(
    seq_len: int,
    regime: str,
    seed: int = 42,
) -> tuple[List[str], MembershipMergeTable, CustomTokenizer]:
    """Generates synthetic token streams and merge tables for controlled testing."""
    rng = random.Random(seed)
    # Characters pool: half words, half prefix spaces
    vocab_atoms = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    pieces: List[str] = []
    for i in range(seq_len):
        atom = rng.choice(vocab_atoms)
        pieces.append(atom if (i % 2 == 0) else S + atom)

    # Construct candidate results according to regime
    results: List[str] = []
    if regime == "sparse":
        # Only ~5% of adjacent pairs can merge
        for i in range(0, seq_len - 1, 20):
            results.append(pieces[i] + pieces[i + 1])
    elif regime == "medium":
        # ~25% of adjacent pairs can merge
        for i in range(0, seq_len - 1, 4):
            results.append(pieces[i] + pieces[i + 1])
    elif regime == "dense":
        # Alternating merges across the stream
        for i in range(0, seq_len - 1, 2):
            results.append(pieces[i] + pieces[i + 1])
    elif regime == "hierarchical":
        # 3-level tree merges: A+B -> AB, C+D -> CD, AB+CD -> ABCD
        for i in range(0, seq_len - 3, 4):
            ab = pieces[i] + pieces[i + 1]
            cd = pieces[i + 2] + pieces[i + 3]
            abcd = ab + cd
            results.extend([ab, cd, abcd])

    unique_results = sorted(set(results))
    all_vocab = sorted(set(pieces + unique_results))
    token_to_id = {tok: idx for idx, tok in enumerate(all_vocab)}

    model = UnigramModel(
        vocab={tok: -1.0 for tok in all_vocab},
        token_to_id=token_to_id,
        id_to_token={i: tok for tok, i in token_to_id.items()},
        special_tokens=[],
        max_subword_len=64,
        byte_fallback=False,
    )
    # Build baseline tokenizer with merge_engine="default" to isolate production
    tok = _create_baseline_benchmark_tokenizer(model)
    # Build table directly from tokenizer's own cross-word definition to guarantee consistency
    table = cross_word_membership_table(tok)
    return pieces, table, tok


def run_benchmark(*, engine: str = "fast") -> List[Dict[str, Any]]:
    ref_engine = ReferenceMergeEngine()
    fast_engine: MergeEngine
    if engine == "native":
        from uniqtoken.native_merge_engine import NativeFastMergeEngine

        fast_engine = NativeFastMergeEngine()
        print("Experimental native prototype; exploratory timings, not production/adoption evidence.")
    elif engine == "fast":
        fast_engine = FastMergeEngine()
    else:
        raise ValueError(f"unknown benchmark engine: {engine!r}")

    test_configs = [
        # (seq_len, regime, iterations, warmup_iters)
        (64, "sparse", 100, 5),
        (64, "dense", 100, 5),
        (256, "sparse", 100, 5),
        (256, "medium", 100, 5),
        (256, "hierarchical", 100, 5),
        (1024, "sparse", 50, 5),
        (1024, "medium", 50, 5),
        (1024, "dense", 50, 5),
        (1024, "hierarchical", 50, 5),
        (4096, "sparse", 25, 5),
        (4096, "medium", 25, 5),
    ]

    records: List[Dict[str, Any]] = []

    print("=" * 105)
    print(
        f"{'Length':<8} | {'Regime':<13} | {'Merges':<7} | {'Iters':<6} | "
        f"{'Ref Mean (ms)':<14} | {'Fast Mean (ms)':<15} | {'Speedup':<8} | {'Parity'}"
    )
    print("-" * 105)

    for seq_len, regime, iters, warmup_iters in test_configs:
        pieces, table, tok = build_synthetic_benchmark_fixture(seq_len, regime)
        constraints = MergeConstraints(
            semantic_profile=SemanticProfile.SUPER_BPE_PASS_V1,
            vocabulary_identity=table.vocabulary_identity,
            hard_cuts=frozenset(),
            legality=None,
        )

        # 1. Warm-up and Full Parity Verification (Oracle vs Fast vs Inlined Production vs Tokenizer Integration)
        ref_out, ref_plan = apply_engine_to_pieces(ref_engine, pieces, table, constraints, None)
        fast_out, fast_plan = apply_engine_to_pieces(fast_engine, pieces, table, constraints, None)
        prod_out = tok._apply_cross_word_merges(list(pieces), 0.0)

        # Select explicitly for this synthetic benchmark; defaults are unchanged.
        tok.set_merge_engine(fast_engine)
        tok_fast_out = tok._apply_cross_word_merges(list(pieces), 0.0)
        tok.set_merge_engine("default")

        parity = ref_out == fast_out == prod_out == tok_fast_out and ref_plan == fast_plan
        if not parity:
            raise RuntimeError(f"Parity mismatch on {seq_len}, {regime}")

        # Warm-up phase
        for _ in range(warmup_iters):
            apply_engine_to_pieces(ref_engine, pieces, table, constraints, None)
            apply_engine_to_pieces(fast_engine, pieces, table, constraints, None)

        # Benchmark Reference Engine
        ref_times: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            apply_engine_to_pieces(ref_engine, pieces, table, constraints, None)
            ref_times.append((time.perf_counter() - t0) * 1000.0)

        # Benchmark Fast Engine
        fast_times: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            apply_engine_to_pieces(fast_engine, pieces, table, constraints, None)
            fast_times.append((time.perf_counter() - t0) * 1000.0)

        ref_mean = statistics.mean(ref_times)
        fast_mean = statistics.mean(fast_times)
        ref_min = min(ref_times)
        fast_min = min(fast_times)
        ref_max = max(ref_times)
        fast_max = max(fast_times)
        ref_std = statistics.stdev(ref_times) if len(ref_times) > 1 else 0.0
        fast_std = statistics.stdev(fast_times) if len(fast_times) > 1 else 0.0

        speedup = ref_mean / max(fast_mean, 1e-6)

        record = {
            "candidate_engine": fast_engine.name,
            "sequence_length": seq_len,
            "regime": regime,
            "applied_merges": ref_plan.applied_merges,
            "iterations": iters,
            "warmup_iterations": warmup_iters,
            "reference_ms": round(ref_mean, 3),
            "fast_ms": round(fast_mean, 3),
            "reference_min_ms": round(ref_min, 3),
            "fast_min_ms": round(fast_min, 3),
            "reference_max_ms": round(ref_max, 3),
            "fast_max_ms": round(fast_max, 3),
            "reference_std_ms": round(ref_std, 3),
            "fast_std_ms": round(fast_std, 3),
            "speedup": round(speedup, 2),
            "parity_verified": parity,
        }
        records.append(record)

        print(
            f"{seq_len:<8} | {regime:<13} | {ref_plan.applied_merges:<7} | {iters:<6} | "
            f"{ref_mean:<14.3f} | {fast_mean:<15.3f} | {speedup:<7.2f}x | "
            f"{'PASS' if parity else 'FAIL'}"
        )

    print("=" * 105)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("fast", "native"), default="fast")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.engine == "native" and args.out is None:
        parser.error("--engine native requires --out to avoid overwriting existing benchmark evidence")
    benchmark_records = run_benchmark(engine=args.engine)
    output_path = args.out if args.out is not None else Path(__file__).parent / "fast_merge_engine_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)
        f.write("\n")
    print(f"Results saved to {output_path}")
