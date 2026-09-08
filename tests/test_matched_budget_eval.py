"""
Tests for Matched-Budget Benchmark Harness (Issue #50).
Verifies corpus generation across 8 domains, analytical FLOP calculations,
tokenizer training, model evaluation metrics, and plot generation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from benchmarks.run_matched_budget_eval import (
    DEFAULT_VOCAB_BUDGETS,
    LM_CONFIGS,
    calculate_analytical_flops_per_step,
    generate_balanced_multilingual_corpus,
    generate_tradeoff_plots,
    run_benchmark,
)


def test_generate_balanced_multilingual_corpus():
    train_docs, val_by_domain = generate_balanced_multilingual_corpus(num_docs_per_lang=10, seed=42)

    required_domains = {"English", "Hindi", "Telugu", "Arabic", "Chinese", "Russian", "Code", "Finnish"}
    assert set(val_by_domain.keys()) == required_domains, (
        f"Missing domains: {required_domains - set(val_by_domain.keys())}"
    )

    assert len(train_docs) > 0
    for domain, text in val_by_domain.items():
        assert len(text) > 0, f"Validation text for {domain} is empty"


def test_analytical_flops_calculation():
    cfg = LM_CONFIGS["Small (2L-128d)"]
    p_total, p_non_embed, flops_per_step = calculate_analytical_flops_per_step(
        vocab_size=8192,
        cfg=cfg,
        seq_len=64,
    )

    assert p_total > p_non_embed
    assert p_non_embed > 0
    assert flops_per_step > 0.0

    # Ensure FLOP scaling with vocabulary size increases embedding and head projection cost
    _, _, flops_larger_vocab = calculate_analytical_flops_per_step(
        vocab_size=16384,
        cfg=cfg,
        seq_len=64,
    )
    assert flops_larger_vocab > flops_per_step


def test_run_benchmark_smoke_execution():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    records, metadata = run_benchmark(
        vocab_budgets=[1024],
        lm_tiers=["Small (2L-128d)"],
        device_str=device_str,
        target_flops=2.0e9,
        num_docs_per_lang=25,
        seed=123,
        verbose=False,
    )

    assert len(records) == 3  # 3 tokenizers x 1 vocab x 1 LM tier
    tok_names = {r.tokenizer_name for r in records}
    assert tok_names == {"SentencePiece-Unigram", "Boundary-BPE", "UniqToken-SuperBPE"}

    for r in records:
        assert r.vocab_budget == 1024
        assert r.lm_tier == "Small (2L-128d)"
        assert r.bytes_per_token > 0.0
        assert r.tokens_per_byte > 0.0
        assert r.token_ce_loss > 0.0
        assert r.true_lm_bpb > 0.0
        assert r.training_steps >= 1
        assert r.actual_flops > 0.0
        assert r.encode_latency_us >= 0.0
        assert r.peak_rss_mb >= 0.0
        if device_str == "cuda":
            assert r.peak_vram_mb >= 0.0

    assert metadata["cuda_available"] == torch.cuda.is_available()


def test_generate_tradeoff_plots_execution():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    records, _ = run_benchmark(
        vocab_budgets=[1024],
        lm_tiers=["Small (2L-128d)"],
        device_str=device_str,
        target_flops=1.0e9,
        num_docs_per_lang=15,
        seed=42,
        verbose=False,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = Path(tmp_dir) / "test_plots"
        png_path, svg_path = generate_tradeoff_plots(records, prefix)

        assert png_path.exists()
        assert svg_path.exists()
        assert png_path.stat().st_size > 0
        assert svg_path.stat().st_size > 0
