"""
Tests for Matched-Budget Benchmark Harness (Issue #50).
Verifies corpus generation across 8 domains, analytical FLOP calculations,
tokenizer training, model evaluation metrics, and plot generation.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import matplotlib

    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None
    HAS_MATPLOTLIB = False

from benchmarks.run_matched_budget_eval import (
    DEFAULT_VOCAB_BUDGETS,
    LM_CONFIGS,
    calculate_analytical_flops_per_step,
    generate_balanced_multilingual_corpus,
    generate_tradeoff_plots,
    run_benchmark,
)


class MatchedBudgetBenchmarkTests(unittest.TestCase):
    def test_generate_balanced_multilingual_corpus(self):
        train_docs, val_by_domain = generate_balanced_multilingual_corpus(num_docs_per_lang=10, seed=42)

        required_domains = {"English", "Hindi", "Telugu", "Arabic", "Chinese", "Russian", "Code", "Finnish"}
        self.assertEqual(
            set(val_by_domain.keys()),
            required_domains,
            f"Missing domains: {required_domains - set(val_by_domain.keys())}",
        )

        self.assertGreater(len(train_docs), 0)
        for domain, text in val_by_domain.items():
            self.assertGreater(len(text), 0, f"Validation text for {domain} is empty")

    def test_generate_1mib_corpus(self):
        train_docs, val_by_domain = generate_balanced_multilingual_corpus(num_docs_per_lang=260, seed=42)
        total_bytes = sum(len(d.encode("utf-8")) for d in train_docs) + sum(
            len(v.encode("utf-8")) for v in val_by_domain.values()
        )
        # Check that corpus size scales to approximately 1 MiB (within 0.85 - 1.25 MiB)
        self.assertTrue(
            0.85 * 1024 * 1024 <= total_bytes <= 1.25 * 1024 * 1024,
            f"Expected approx 1 MiB corpus, got {total_bytes / (1024 * 1024):.2f} MiB",
        )

    def test_analytical_flops_calculation(self):
        cfg = LM_CONFIGS["Small (2L-128d)"]
        p_total, p_non_embed, flops_per_step = calculate_analytical_flops_per_step(
            vocab_size=8192,
            cfg=cfg,
            seq_len=64,
        )

        self.assertGreater(p_total, p_non_embed)
        self.assertGreater(p_non_embed, 0)
        self.assertGreater(flops_per_step, 0.0)

        # Ensure FLOP scaling with vocabulary size increases embedding and head projection cost
        _, _, flops_larger_vocab = calculate_analytical_flops_per_step(
            vocab_size=16384,
            cfg=cfg,
            seq_len=64,
        )
        self.assertGreater(flops_larger_vocab, flops_per_step)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required for Transformer LM benchmarks")
    def test_run_benchmark_smoke_execution(self):
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

        self.assertEqual(len(records), 3)  # 3 tokenizers x 1 vocab x 1 LM tier
        tok_names = {r.tokenizer_name for r in records}
        self.assertEqual(tok_names, {"SentencePiece-Unigram", "Boundary-BPE", "UniqToken-SuperBPE"})

        for r in records:
            self.assertEqual(r.vocab_budget, 1024)
            self.assertEqual(r.lm_tier, "Small (2L-128d)")
            self.assertGreater(r.bytes_per_token, 0.0)
            self.assertGreater(r.tokens_per_byte, 0.0)
            self.assertGreater(r.token_ce_loss, 0.0)
            self.assertGreater(r.true_lm_bpb, 0.0)
            self.assertGreaterEqual(r.training_steps, 1)
            self.assertGreater(r.actual_flops, 0.0)
            self.assertGreaterEqual(r.encode_latency_us, 0.0)
            self.assertGreaterEqual(r.peak_rss_mb, 0.0)
            if device_str == "cuda":
                self.assertGreater(r.peak_vram_mb, 0.0)

        self.assertEqual(metadata["cuda_available"], torch.cuda.is_available())

    @unittest.skipUnless(HAS_TORCH and HAS_MATPLOTLIB, "PyTorch and matplotlib required for plot tests")
    def test_generate_tradeoff_plots_execution(self):
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

            self.assertTrue(png_path.exists())
            self.assertTrue(svg_path.exists())
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(svg_path.stat().st_size, 0)
