"""
Throughput Benchmark: Single-String vs Rayon Batch vs Production Baselines.

Evaluates raw tokenization throughput across:
- UniqToken (Single-String Rust Viterbi)
- UniqToken (Rayon Multi-Threaded Batch)
- SentencePiece (C++)
- HuggingFace Tokenizers (Rust Fast Tokenizer)
- tiktoken (Rust)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uniqtoken.batch_collator import BatchCollator
from uniqtoken.tokenizer import CustomTokenizer


def run_throughput_benchmark(num_sentences: int = 5000) -> None:
    corpus = [
        "The transformer architecture relies on subword tokenization to compress sequence length.",
        "Exact offset alignment is essential for accurate span extraction and structured decoding.",
        "High performance native Rust modules allow multi-threaded parallel batch execution without GIL lock.",
        "Neural language modeling balances vocabulary size against computational embedding cost.",
    ]
    texts = corpus * (num_sentences // len(corpus))

    print("=" * 105)
    print(
        f"UNIQTOKEN THROUGHPUT BENCHMARK (Workload: {len(texts):,} sentences, ~{len(' '.join(texts).encode('utf-8')) / (1024 * 1024):.2f} MB text)"
    )
    print("=" * 105)

    # 1. Train 1K UniqToken Model
    tok = CustomTokenizer.train_from_corpus(
        corpus=corpus * 50,
        target_vocab_size=1000,
        ranking_strategy="char_savings",
        min_frequency=1,
        verbose=False,
    )
    collator = BatchCollator(tok)

    # Measure UniqToken Single-String Python -> Rust Dispatch
    t0 = time.perf_counter()
    single_tokens_count = 0
    for text in texts:
        single_tokens_count += len(tok.encode(text))
    t_single = time.perf_counter() - t0
    rate_single = single_tokens_count / max(t_single, 1e-6)

    # Measure UniqToken Collator Batch (Python Normalization + Rayon Spans)
    t0 = time.perf_counter()
    batch_enc = collator.batch_encode(texts, padding=False, add_special_tokens=False)
    t_batch = time.perf_counter() - t0
    batch_tokens_count = sum(len(seq) for seq in batch_enc.input_ids)
    rate_batch = batch_tokens_count / max(t_batch, 1e-6)

    # Measure UniqToken Fused Native Pipeline (one FFI: normalize+pretokenize+Viterbi+IDs)
    t0 = time.perf_counter()
    native_ids = tok.encode_to_ids_batch(texts)
    t_native = time.perf_counter() - t0
    native_tokens_count = sum(len(seq) for seq in native_ids)
    rate_native = native_tokens_count / max(t_native, 1e-6)

    # Measure UniqToken Pure Native Rayon Batch (Zero-Copy Integer Stream)
    rate_raw_rayon = 0.0
    raw_rayon_count = 0
    t_raw_rayon = 0.0
    try:
        import uniqtoken_core
    except ImportError:
        uniqtoken_core = None

    try:
        rust_trie = tok.model._get_rust_trie()
        if rust_trie is not None and uniqtoken_core is not None:
            # rust_encode_ids_batch expects pre-tokenized chunks (exactly what
            # tok.encode feeds the model) — NOT raw sentences. Include the
            # Python normalize/pre-tokenize time in the reported rate.
            t0 = time.perf_counter()
            flat_chunks: List[str] = []
            for t in texts:
                flat_chunks.extend(tok.pre_tokenizer.pre_tokenize(tok.normalizer.normalize(t)))
            t_prep = time.perf_counter() - t0
            t0 = time.perf_counter()
            chunk_ids = uniqtoken_core.rust_encode_ids_batch(flat_chunks, rust_trie, tok.model.byte_fallback)
            t_enc = time.perf_counter() - t0
            t_raw_rayon = t_prep + t_enc
            raw_rayon_count = sum(len(x) for x in chunk_ids)
            rate_raw_rayon = raw_rayon_count / max(t_raw_rayon, 1e-6)
    except Exception as e:
        print(f"Raw Rayon error: {e}")

    # Measure SentencePiece (C++)
    rate_sp = 0.0
    sp_tokens_count = 0
    t_sp = 0.0
    try:
        import sentencepiece as spm

        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            sp_corpus = tmp / "train.txt"
            sp_corpus.write_text("\n".join(corpus * 50), encoding="utf-8")
            sp_prefix = tmp / "sp_model"
            spm.SentencePieceTrainer.train(
                input=str(sp_corpus),
                model_prefix=str(sp_prefix),
                model_type="unigram",
                vocab_size=1000,
                character_coverage=1.0,
                byte_fallback=True,
                hard_vocab_limit=False,
                minloglevel=2,
            )
            sp_proc = spm.SentencePieceProcessor(model_file=str(sp_prefix) + ".model")
            t0 = time.perf_counter()
            sp_res = sp_proc.encode(texts, out_type=int)
            t_sp = time.perf_counter() - t0
            sp_tokens_count = sum(len(x) for x in sp_res)
            rate_sp = sp_tokens_count / max(t_sp, 1e-6)
    except Exception as e:
        print(f"SentencePiece baseline error: {e}")

    # Measure Hugging Face Fast Tokenizers (Rust)
    rate_hf = 0.0
    hf_tokens_count = 0
    t_hf = 0.0
    try:
        import json
        from tokenizers import Tokenizer
        from uniqtoken.hf_exporter import HuggingFaceExporter

        with TemporaryDirectory() as tmp_dir:
            hf_dict = HuggingFaceExporter.export_to_hf_dict(tok)
            path = Path(tmp_dir) / "hf_tok.json"
            path.write_text(json.dumps(hf_dict), encoding="utf-8")
            hf_tok = Tokenizer.from_file(str(path))
            t0 = time.perf_counter()
            hf_res = hf_tok.encode_batch(texts)
            t_hf = time.perf_counter() - t0
            hf_tokens_count = sum(len(x.ids) for x in hf_res)
            rate_hf = hf_tokens_count / max(t_hf, 1e-6)
    except Exception as e:
        print(f"HuggingFace baseline error: {e}")

    # Measure tiktoken (Rust)
    rate_tiktoken = 0.0
    tiktoken_count = 0
    t_tiktoken = 0.0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        t0 = time.perf_counter()
        res_tt = enc.encode_batch(texts)
        t_tiktoken = time.perf_counter() - t0
        tiktoken_count = sum(len(x) for x in res_tt)
        rate_tiktoken = tiktoken_count / max(t_tiktoken, 1e-6)
    except Exception as e:
        print(f"tiktoken baseline error: {e}")

    total_input_bytes = sum(len(t.encode("utf-8")) for t in texts)
    mb_total = total_input_bytes / (1024 * 1024)

    # Compute MB/s rates
    mb_s_single = mb_total / max(t_single, 1e-6)
    mb_s_batch = mb_total / max(t_batch, 1e-6)
    mb_s_native = mb_total / max(t_native, 1e-6) if t_native > 0 else 0.0
    mb_s_raw = mb_total / max(t_raw_rayon, 1e-6) if t_raw_rayon > 0 else 0.0
    mb_s_hf = mb_total / max(t_hf, 1e-6) if t_hf > 0 else 0.0
    mb_s_sp = mb_total / max(t_sp, 1e-6) if t_sp > 0 else 0.0
    mb_s_tt = mb_total / max(t_tiktoken, 1e-6) if t_tiktoken > 0 else 0.0

    # Output Clean Comparison Table
    hdr = f"{'Tokenizer Engine':<30} | {'Tokens':<8} | {'B/Tok':<6} | {'Time (s)':<9} | {'Tok/sec':<16} | {'MB/sec':<12} | {'Speedup':<10}"
    print(hdr)
    print("-" * len(hdr))
    print(
        f"{'UniqToken (Single Python Dispatch)':<30} | {single_tokens_count:<8} | {total_input_bytes / max(single_tokens_count, 1):<6.2f} | {t_single:<9.4f} | {rate_single:>12,.0f} tok/s | {mb_s_single:>8.2f} MB/s | {'1.00x':<10}"
    )
    print(
        f"{'UniqToken (Collator + Rayon Spans)':<30} | {batch_tokens_count:<8} | {total_input_bytes / max(batch_tokens_count, 1):<6.2f} | {t_batch:<9.4f} | {rate_batch:>12,.0f} tok/s | {mb_s_batch:>8.2f} MB/s | {f'{rate_batch / max(rate_single, 1e-6):.2f}x':<10}"
    )
    if rate_native > 0:
        print(
            f"{'UniqToken (Fused Native Pipeline)':<30} | {native_tokens_count:<8} | {total_input_bytes / max(native_tokens_count, 1):<6.2f} | {t_native:<9.4f} | {rate_native:>12,.0f} tok/s | {mb_s_native:>8.2f} MB/s | {f'{rate_native / max(rate_single, 1e-6):.2f}x':<10}"
        )
    if rate_raw_rayon > 0:
        print(
            f"{'UniqToken (Rayon Parallel Stream)':<30} | {raw_rayon_count:<8} | {total_input_bytes / max(raw_rayon_count, 1):<6.2f} | {t_raw_rayon:<9.4f} | {rate_raw_rayon:>12,.0f} tok/s | {mb_s_raw:>8.2f} MB/s | {f'{rate_raw_rayon / max(rate_single, 1e-6):.2f}x':<10}"
        )
    if rate_hf > 0:
        print(
            f"{'HuggingFace Tokenizers (Rust)':<30} | {hf_tokens_count:<8} | {total_input_bytes / max(hf_tokens_count, 1):<6.2f} | {t_hf:<9.4f} | {rate_hf:>12,.0f} tok/s | {mb_s_hf:>8.2f} MB/s | {f'{rate_hf / max(rate_single, 1e-6):.2f}x':<10}"
        )
    if rate_sp > 0:
        print(
            f"{'SentencePiece (C++ Batch)':<30} | {sp_tokens_count:<8} | {total_input_bytes / max(sp_tokens_count, 1):<6.2f} | {t_sp:<9.4f} | {rate_sp:>12,.0f} tok/s | {mb_s_sp:>8.2f} MB/s | {f'{rate_sp / max(rate_single, 1e-6):.2f}x':<10}"
        )
    if rate_tiktoken > 0:
        print(
            f"{'tiktoken (cl100k_base Rust)':<30} | {tiktoken_count:<8} | {total_input_bytes / max(tiktoken_count, 1):<6.2f} | {t_tiktoken:<9.4f} | {rate_tiktoken:>12,.0f} tok/s | {mb_s_tt:>8.2f} MB/s | {f'{rate_tiktoken / max(rate_single, 1e-6):.2f}x':<10}"
        )
    print("=" * 105 + "\n")


if __name__ == "__main__":
    run_throughput_benchmark(num_sentences=10000)
