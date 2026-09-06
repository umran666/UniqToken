"""
Downstream LLM Model Pretraining Validation Benchmark.

Evaluates tokenizer efficiency in end-to-end Transformer Language Model training:
1. Trains identical architecture MiniTransformerLM models across tokenizer variants:
   - UniqToken Unigram
   - UniqToken SuperBPE
   - Standard BPE
2. Measures:
   - Validation Cross-Entropy Loss
   - Bits-Per-Byte (BPB): Loss * Tokens / (Bytes * ln(2))
   - Effective context window utilization
   - Training step throughput (tokens/sec and bytes/sec)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uniqtoken.bpe_trainer as bpe_trainer
from uniqtoken.cem_merger import CrossEntropyMerging
from uniqtoken.tokenizer import CustomTokenizer


@dataclass
class PretrainingMetrics:
    model_name: str
    vocab_size: int
    total_tokens: int
    total_bytes: int
    evaluated_tokens: int
    evaluated_bytes: int
    compression_ratio: float  # bytes per token
    final_loss: float
    bits_per_byte: float
    tokens_per_sec: float
    bytes_per_sec: float


# Synthetic multilingual and code corpus for reproducible downstream training
PRETRAINING_CORPUS = [
    "def compute_gradients(loss, params, lr=1e-3):\n    for p in params:\n        p.data -= lr * p.grad\n    return params",
    "class MultiHeadAttention(nn.Module):\n    def __init__(self, dim, heads):\n        super().__init__()\n        self.qkv = nn.Linear(dim, dim * 3)\n    def forward(self, x):\n        return self.qkv(x)",
    "प्राकृतिक भाषा प्रसंस्करण और गहन शिक्षण एल्गोरिदम मॉडल को सशक्त बनाते हैं।",
    "日本語の自然言語処理と機械学習モデルの訓練において、トークナイザーの圧縮率は極めて重要です。",
    "معالجة اللغة الطبيعية والذكاء الاصطناعي يتطلبان تمثيلاً فعالاً للنصوص.",
    "for i in range(100):\n    x = torch.randn(32, 128)\n    loss = (x ** 2).mean()\n    loss.backward()",
    "Transformer language models optimize cross-entropy loss over subword tokens.",
    "Data compression directly impacts the effective context window capacity of large models.",
] * 32


class BPETokenizerAdapter:
    def __init__(
        self,
        model: bpe_trainer.BPEModel,
        normalizer: Any = None,
        pre_tokenizer: Any = None,
    ):
        self.model = model
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    def encode_to_ids(self, text: str) -> List[int]:
        if self.normalizer is None or self.pre_tokenizer is None:
            chunks = [text]
        else:
            norm = self.normalizer.normalize(text)
            chunks = self.pre_tokenizer.pre_tokenize(norm)
        unk_id = self.model.token_to_id.get("<|unk|>", 0)
        ids: List[int] = []
        for chunk in chunks:
            ids.extend(self.model.token_to_id.get(t, unk_id) for t in self.model.encode(chunk))
        return ids

    def decode(self, token_ids: List[int]) -> str:
        return self.model.decode(token_ids)


def create_tokenizers(target_vocab: int = 500, corpus: Optional[List[str]] = None) -> Dict[str, Any]:
    """Builds and returns calibrated tokenizers for downstream comparison."""
    tokenizers: Dict[str, Any] = {}
    training_corpus = list(corpus if corpus is not None else PRETRAINING_CORPUS)

    # 1. UniqToken Unigram
    unigram_tok = CustomTokenizer.train_from_corpus(
        corpus=training_corpus,
        target_vocab_size=target_vocab,
        ranking_strategy="pmi",
        min_frequency=1,
        verbose=False,
    )
    tokenizers["UniqToken (Unigram)"] = unigram_tok

    # 2. UniqToken SuperBPE
    pretok_chunks: List[str] = []
    for doc in training_corpus:
        norm = unigram_tok.normalizer.normalize(doc)
        pretok_chunks.extend(unigram_tok.pre_tokenizer.pre_tokenize(norm))
    cem = CrossEntropyMerging(max_merges=30, cross_word=True, verbose=False)
    sbp_model = cem.optimize(unigram_tok.model, chunks=pretok_chunks)
    tokenizers["UniqToken (SuperBPE)"] = CustomTokenizer(
        normalizer=unigram_tok.normalizer,
        pre_tokenizer=unigram_tok.pre_tokenizer,
        model=sbp_model,
    )

    # 3. Standard BPE — trained and applied on the same pre-tokenized chunks
    #    as the UniqToken variants so the baseline is directly comparable.
    bpe_chunks: List[str] = []
    for doc in training_corpus:
        norm = unigram_tok.normalizer.normalize(doc)
        bpe_chunks.extend(unigram_tok.pre_tokenizer.pre_tokenize(norm))
    bpe_trainer_inst = bpe_trainer.BPETrainer(
        target_vocab_size=target_vocab,
        byte_fallback=True,
    )
    bpe_model = bpe_trainer_inst.train(bpe_chunks, verbose=False)
    tokenizers["Standard BPE"] = BPETokenizerAdapter(
        bpe_model,
        normalizer=unigram_tok.normalizer,
        pre_tokenizer=unigram_tok.pre_tokenizer,
    )

    return tokenizers


def _split_documents(corpus: List[str]) -> Tuple[List[str], List[str]]:
    """Splits by document identity so exact duplicates cannot cross the boundary."""
    unique_documents = list(dict.fromkeys(corpus))
    if len(unique_documents) < 2:
        return list(corpus), list(corpus)
    validation_size = max(1, len(unique_documents) // 5)
    validation_documents = set(unique_documents[-validation_size:])
    train_documents = [doc for doc in corpus if doc not in validation_documents]
    held_out_documents = [doc for doc in corpus if doc in validation_documents]
    return train_documents, held_out_documents


def train_toy_transformer(
    tok: Any,
    model_label: str,
    corpus: List[str],
    steps: int = 40,
    seq_len: int = 48,
    batch_size: int = 4,
    dim: int = 64,
    heads: int = 4,
    layers: int = 2,
    device: str = "auto",
    seed: int = 42,
) -> PretrainingMetrics:
    """
    Trains a causal mini-transformer or lightweight probabilistic model
    and measures cross-entropy loss and bits-per-byte (BPB).

    ``device`` selects the compute device: ``"auto"`` (default) uses
    CUDA when ``torch.cuda.is_available()``, otherwise CPU. Pass
    ``"cuda"`` or ``"cpu"`` to override.
    """
    if not corpus or not any(corpus):
        raise ValueError("corpus must contain at least one non-empty document")
    if steps < 1 or seq_len < 1 or batch_size < 1:
        raise ValueError("steps, seq_len, and batch_size must be positive")
    if dim < 1 or heads < 1 or layers < 1 or dim % heads != 0:
        raise ValueError("dim, heads, and layers must be positive and dim must be divisible by heads")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be 'auto', 'cpu', or 'cuda' (got {device!r})")

    # Keep exact duplicate documents entirely on one side of the split.
    train_docs, val_docs = _split_documents(corpus)

    def _flatten_ids(docs: List[str]) -> List[int]:
        flat: List[int] = []
        for doc in docs:
            flat.extend(tok.encode_to_ids(doc))
        return flat

    flat_tokens = _flatten_ids(corpus)
    train_flat = _flatten_ids(train_docs)
    val_flat = _flatten_ids(val_docs)

    total_tokens = len(flat_tokens)
    total_bytes = sum(len(doc.encode("utf-8")) for doc in corpus)
    validation_bytes = sum(len(doc.encode("utf-8")) for doc in val_docs)
    if total_tokens == 0 or total_bytes == 0 or not val_flat or validation_bytes == 0:
        raise ValueError("corpus must produce non-empty token and byte sequences")
    compression = total_bytes / total_tokens
    token_to_id = getattr(getattr(tok, "model", None), "token_to_id", None)
    max_token_id = max(token_to_id.values(), default=-1) if token_to_id else -1
    vocab_size = max(int(tok.vocab_size), max_token_id + 1)

    # 2. Check PyTorch availability
    has_torch = False
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        has_torch = True
    except ImportError:
        pass

    if has_torch and len(train_flat) > seq_len:
        import torch
        import torch.nn as nn

        class MiniCausalLM(nn.Module):
            def __init__(self, vs: int, d: int, h: int, n_l: int, max_s: int):
                super().__init__()
                self.tok_emb = nn.Embedding(vs, d)
                self.pos_emb = nn.Embedding(max_s, d)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=h,
                    dim_feedforward=d * 2,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                )
                self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_l)
                self.norm = nn.LayerNorm(d)
                self.head = nn.Linear(d, vs, bias=False)
                self.head.weight = self.tok_emb.weight  # Weight tying

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, s = x.size()
                pos = torch.arange(0, s, device=x.device).unsqueeze(0)
                h = self.tok_emb(x) + self.pos_emb(pos)
                causal_mask = torch.triu(torch.full((s, s), float("-inf"), device=x.device), diagonal=1)
                out = self.blocks(h, mask=causal_mask)
                out = self.norm(out)
                return self.head(out)

        target_device: torch.device
        if device == "auto":
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
            target_device = torch.device(device)
        torch.manual_seed(seed)
        model = MiniCausalLM(vs=vocab_size, d=dim, h=heads, n_l=layers, max_s=seq_len).to(target_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        start_time = time.perf_counter()

        data_tensor = torch.tensor(train_flat, dtype=torch.long)
        max_start = len(train_flat) - seq_len - 1

        last_train_loss = 0.0
        model.train()
        for step in range(steps):
            optimizer.zero_grad()
            # Sample batch (each row is a distinct contiguous window)
            batch_inputs = []
            batch_targets = []
            for b in range(batch_size):
                idx = (step * batch_size + b) % (max_start + 1)
                chunk = data_tensor[idx : idx + seq_len + 1]
                batch_inputs.append(chunk[:-1])
                batch_targets.append(chunk[1:])

            inputs = torch.stack(batch_inputs).to(target_device)
            targets = torch.stack(batch_targets).to(target_device)

            logits = model(inputs)
            loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            last_train_loss = float(loss.item())

        # Evaluate on held-out validation windows
        model.eval()
        final_loss = last_train_loss
        evaluated_tokens = 0
        if len(val_flat) >= 2:
            val_tensor = torch.tensor(val_flat, dtype=torch.long)
            weighted_validation_loss = 0.0
            with torch.no_grad():
                for start in range(0, len(val_flat) - 1, seq_len):
                    chunk = val_tensor[start : start + seq_len + 1].to(target_device)
                    prediction_count = len(chunk) - 1
                    if prediction_count == 0:
                        continue
                    vlogits = model(chunk[:-1].unsqueeze(0))
                    vloss = loss_fn(vlogits.view(-1, vocab_size), chunk[1:].view(-1))
                    weighted_validation_loss += float(vloss.item()) * prediction_count
                    evaluated_tokens += prediction_count
            if evaluated_tokens:
                final_loss = weighted_validation_loss / evaluated_tokens
        processed_tokens = steps * batch_size * seq_len
    else:
        # Fit a Laplace-smoothed unigram model on training data and evaluate it
        # only on held-out tokens. This is a real held-out baseline, not entropy
        # estimated from the validation distribution itself.
        start_time = time.perf_counter()
        train_counts: Dict[int, int] = {}
        for token_id in train_flat:
            train_counts[token_id] = train_counts.get(token_id, 0) + 1
        denominator = len(train_flat) + vocab_size
        final_loss = sum(-math.log((train_counts.get(token_id, 0) + 1) / denominator) for token_id in val_flat) / len(
            val_flat
        )
        evaluated_tokens = len(val_flat)
        processed_tokens = len(train_flat) + len(val_flat)

    elapsed = max(time.perf_counter() - start_time, 1e-6)
    tok_per_sec = processed_tokens / elapsed
    bytes_per_sec = processed_tokens * compression / elapsed

    if evaluated_tokens == 0:
        raise ValueError("validation data must contain at least one predictable token")
    # Use the same held-out population for loss and byte accounting.
    bits_per_byte = (final_loss * evaluated_tokens) / (validation_bytes * math.log(2.0))

    return PretrainingMetrics(
        model_name=model_label,
        vocab_size=vocab_size,
        total_tokens=total_tokens,
        total_bytes=total_bytes,
        evaluated_tokens=evaluated_tokens,
        evaluated_bytes=validation_bytes,
        compression_ratio=compression,
        final_loss=final_loss,
        bits_per_byte=bits_per_byte,
        tokens_per_sec=tok_per_sec,
        bytes_per_sec=bytes_per_sec,
    )


def run_pretraining_benchmark(steps: int = 40, export_json: Optional[str] = None) -> List[PretrainingMetrics]:
    """Runs downstream mini-transformer pretraining benchmark across tokenizers."""
    train_docs, _ = _split_documents(PRETRAINING_CORPUS)
    tokenizers = create_tokenizers(target_vocab=500, corpus=train_docs)
    results: List[PretrainingMetrics] = []

    print("\n" + "=" * 110)
    print("DOWNSTREAM TRANSFORMER PRETRAINING & BITS-PER-BYTE (BPB) CONVERGENCE")
    print("=" * 110)
    header = f"{'Tokenizer':<24} | {'Vocab':<6} | {'Tokens':<7} | {'Bytes/Tok':<10} | {'Loss':<8} | {'Bits/Byte (BPB)':<16} | {'Tok/Sec':<10}"
    print(header)
    print("-" * 110)

    for name, tok in tokenizers.items():
        metrics = train_toy_transformer(tok, name, PRETRAINING_CORPUS, steps=steps)
        results.append(metrics)
        row = (
            f"{metrics.model_name:<24} | "
            f"{metrics.vocab_size:<6} | "
            f"{metrics.total_tokens:<7} | "
            f"{metrics.compression_ratio:<10.3f} | "
            f"{metrics.final_loss:<8.4f} | "
            f"{metrics.bits_per_byte:<16.4f} | "
            f"{metrics.tokens_per_sec:<10.1f}"
        )
        print(row)

    print("=" * 110 + "\n")

    if export_json:
        payload = [
            {
                "model_name": m.model_name,
                "vocab_size": m.vocab_size,
                "total_tokens": m.total_tokens,
                "total_bytes": m.total_bytes,
                "evaluated_tokens": m.evaluated_tokens,
                "evaluated_bytes": m.evaluated_bytes,
                "bytes_per_token": m.compression_ratio,
                "final_loss": m.final_loss,
                "bits_per_byte": m.bits_per_byte,
                "tokens_per_sec": m.tokens_per_sec,
            }
            for m in results
        ]
        with open(export_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Exporter] Saved pretraining benchmark report to: {export_json}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Downstream LLM Pretraining Benchmark")
    parser.add_argument("--steps", type=int, default=40, help="Training steps (default: 40)")
    parser.add_argument("--export-json", type=str, default=None, help="Save metrics as JSON")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be a positive integer")

    run_pretraining_benchmark(steps=args.steps, export_json=args.export_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
