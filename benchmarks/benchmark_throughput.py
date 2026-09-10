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
from itertools import cycle, islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uniqtoken.batch_collator import BatchCollator
from uniqtoken.tokenizer import CustomTokenizer

from benchmarks.benchmark_suite import TokenizerBenchmarkSuite


def _train_sentencepiece(spm, training_corpus: List[str], target_vocab: int):
    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        sp_corpus = tmp / "train.txt"
        sp_corpus.write_text("\n".join(training_corpus), encoding="utf-8")
        sp_prefix = tmp / "sp_model"
        spm.SentencePieceTrainer.train(
            input=str(sp_corpus),
            model_prefix=str(sp_prefix),
            model_type="unigram",
            vocab_size=target_vocab,
            character_coverage=1.0,
            byte_fallback=True,
            hard_vocab_limit=True,
            minloglevel=2,
        )
        processor = spm.SentencePieceProcessor(model_file=str(sp_prefix) + ".model")
        if processor.get_piece_size() != target_vocab:
            raise RuntimeError(
                f"SentencePiece produced {processor.get_piece_size()} pieces; requested budget is {target_vocab}"
            )
        return processor


def run_throughput_benchmark(num_sentences: int = 5000) -> None:
    if num_sentences < 1:
        raise ValueError("num_sentences must be positive")
    training_corpus = list(TokenizerBenchmarkSuite.TRAINING_CORPUS)
    evaluation_corpus = [
        "The transformer architecture relies on subword tokenization to compress sequence length.",
        "Exact offset alignment is essential for accurate span extraction and structured decoding.",
        "High performance native Rust modules allow multi-threaded parallel batch execution without GIL lock.",
        "Neural language modeling balances vocabulary size against computational embedding cost.",
    ]
    texts = list(islice(cycle(evaluation_corpus), num_sentences))

    print("=" * 105)
    print(
        f"UNIQTOKEN THROUGHPUT BENCHMARK (Workload: {len(texts):,} sentences, ~{len(' '.join(texts).encode('utf-8')) / (1024 * 1024):.2f} MB text)"
    )
    print("=" * 105)

    # 1. Train 1K UniqToken Model
    tok = CustomTokenizer.train_from_corpus(
        corpus=training_corpus,
        target_vocab_size=1000,
        min_edge_log_prob=float("-inf"),
        ranking_strategy="char_savings",
        min_frequency=1,
        verbose=False,
    )
    if tok.vocab_size != 1000:
        raise RuntimeError(f"UniqToken produced {tok.vocab_size} pieces; requested budget is 1000")
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

    # Measure SentencePiece (C++)
    rate_sp = 0.0
    sp_tokens_count = 0
    t_sp = 0.0
    try:
        import sentencepiece as spm
    except ImportError:
        spm = None
    if spm is not None:
        sp_proc = _train_sentencepiece(spm, training_corpus, target_vocab=1000)
        t0 = time.perf_counter()
        sp_res = sp_proc.encode(texts, out_type=int)
        t_sp = time.perf_counter() - t0
        sp_tokens_count = sum(len(x) for x in sp_res)
        rate_sp = sp_tokens_count / max(t_sp, 1e-6)

    # Measure Hugging Face Fast Tokenizers (Rust)
    rate_hf = 0.0
    hf_tokens_count = 0
    t_hf = 0.0
    try:
        import json
        from tokenizers import Tokenizer
        from uniqtoken.hf_exporter import HuggingFaceExporter
    except ImportError:
        Tokenizer = None
    if Tokenizer is not None:
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
    # Measure tiktoken (Rust)
    rate_tiktoken = 0.0
    tiktoken_count = 0
    t_tiktoken = 0.0
    try:
        import tiktoken
    except ImportError:
        tiktoken = None
    if tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        t0 = time.perf_counter()
        res_tt = enc.encode_batch(texts)
        t_tiktoken = time.perf_counter() - t0
        tiktoken_count = sum(len(x) for x in res_tt)
        rate_tiktoken = tiktoken_count / max(t_tiktoken, 1e-6)
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
    hdr = f"{'Tokenizer Engine':<30} | {'Tokens':<8} | {'B/Tok':<6} | {'Time (s)':<9} | {'MB/sec':<12} | {'Tok/sec':<16} | {'Uniq impl speedup':<18}"
    print(hdr)
    print("-" * len(hdr))
    print(
        f"{'UniqToken (Single Python Dispatch)':<30} | {single_tokens_count:<8} | {total_input_bytes / max(single_tokens_count, 1):<6.2f} | {t_single:<9.4f} | {mb_s_single:>8.2f} MB/s | {rate_single:>12,.0f} tok/s | {'1.00x':<18}"
    )
    print(
        f"{'UniqToken (Collator + Rayon Spans)':<30} | {batch_tokens_count:<8} | {total_input_bytes / max(batch_tokens_count, 1):<6.2f} | {t_batch:<9.4f} | {mb_s_batch:>8.2f} MB/s | {rate_batch:>12,.0f} tok/s | {f'{mb_s_batch / max(mb_s_single, 1e-6):.2f}x':<18}"
    )
    if rate_native > 0:
        print(
            f"{'UniqToken (Fused Native Pipeline)':<30} | {native_tokens_count:<8} | {total_input_bytes / max(native_tokens_count, 1):<6.2f} | {t_native:<9.4f} | {mb_s_native:>8.2f} MB/s | {rate_native:>12,.0f} tok/s | {f'{mb_s_native / max(mb_s_single, 1e-6):.2f}x':<18}"
        )
    if rate_raw_rayon > 0:
        print(
            f"{'UniqToken (Rayon Parallel Stream)':<30} | {raw_rayon_count:<8} | {total_input_bytes / max(raw_rayon_count, 1):<6.2f} | {t_raw_rayon:<9.4f} | {mb_s_raw:>8.2f} MB/s | {rate_raw_rayon:>12,.0f} tok/s | {f'{mb_s_raw / max(mb_s_single, 1e-6):.2f}x':<18}"
        )
    if rate_hf > 0:
        print(
            f"{'HuggingFace Tokenizers (Rust)':<30} | {hf_tokens_count:<8} | {total_input_bytes / max(hf_tokens_count, 1):<6.2f} | {t_hf:<9.4f} | {mb_s_hf:>8.2f} MB/s | {rate_hf:>12,.0f} tok/s | {'n/a':<18}"
        )
    if rate_sp > 0:
        print(
            f"{'SentencePiece (C++ Batch)':<30} | {sp_tokens_count:<8} | {total_input_bytes / max(sp_tokens_count, 1):<6.2f} | {t_sp:<9.4f} | {mb_s_sp:>8.2f} MB/s | {rate_sp:>12,.0f} tok/s | {'n/a':<18}"
        )
    if rate_tiktoken > 0:
        print(
            f"{'tiktoken (cl100k_base Rust)':<30} | {tiktoken_count:<8} | {total_input_bytes / max(tiktoken_count, 1):<6.2f} | {t_tiktoken:<9.4f} | {mb_s_tt:>8.2f} MB/s | {rate_tiktoken:>12,.0f} tok/s | {'n/a':<18}"
        )
    print("Cross-tokenizer throughput is compared in input MB/s; token/s depends on each tokenizer's segmentation.")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    run_throughput_benchmark(num_sentences=10000)
