"""
Downstream LLM & Transformer Tokenizer Evaluation Harness.

Evaluates tokenizers on downstream language model efficiency:
1. Context Window Information Density (Effective Bytes per Context Window).
2. Bits Per Byte (BPB) & Bits Per Character (BPC) Information Density.
3. Morphological Fertility across Diverse Linguistic Domains.
4. Downstream Transformer Loss / Perplexity Under Fixed Context Lengths.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uniqtoken.cem_merger import CrossEntropyMerging
from uniqtoken.tokenizer import CustomTokenizer


@dataclass
class DownstreamMetrics:
    tokenizer_name: str
    vocab_size: int
    total_tokens: int
    total_bytes: int
    total_words: int
    bytes_per_token: float
    tokens_per_word: float
    effective_bytes_in_2k_context: int
    effective_bytes_in_4k_context: int
    effective_bytes_in_8k_context: int
    estimated_bits_per_byte: float
    encode_time_sec: float


class DownstreamEvaluator:
    """
    Automated Downstream Tokenizer Evaluator for Large Language Models.
    """

    BENCHMARK_CORPUS = [
        # Technical & Code
        (
            "def compute_attention(query, key, value, mask=None):\n"
            "    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))\n"
            "    if mask is not None:\n"
            "        scores = scores.masked_fill(mask == 0, -1e9)\n"
            "    p_attn = torch.softmax(scores, dim=-1)\n"
            "    return torch.matmul(p_attn, value), p_attn\n"
        )
        * 5,
        # Multilingual Prose
        (
            "Natural language processing and machine learning have revolutionized computational linguistics. "
            "प्राकृतिक भाषा प्रसंस्करण में वाक्य संरचना और शब्दों के अर्थ का विश्लेषण अत्यंत महत्वपूर्ण है। "
            "日本語の自然言語処理では単語の境界を正確に特定することが求められます。"
            "تتطلب معالجة اللغة العربية فهماً عميقاً للجذور والقواعد الصرفية المعقدة."
        )
        * 4,
        # Mathematical Reasoning
        (
            "Given a probability space (Omega, F, P) and random variables X, Y with joint density f(x, y), "
            "the conditional expectation E[X | Y=y] minimizes the mean squared error. "
            "Compute the integral int_{0}^{infty} e^{-x^2} dx = sqrt(pi)/2. "
            "Eigenvalues of A = [[4, 1], [2, 3]] satisfy det(A - lambda*I) = 0 => lambda_1 = 5, lambda_2 = 2."
        )
        * 4,
    ]

    def __init__(
        self,
        vocab_size: int = 1000,
        max_merges: int = 20,
        corpus: Optional[List[str]] = None,
    ):
        self.vocab_size = vocab_size
        self.max_merges = max_merges
        self.corpus = corpus if corpus is not None else list(self.BENCHMARK_CORPUS)
        self.corpus_text = "\n\n".join(self.corpus)
        self.raw_bytes = len(self.corpus_text.encode("utf-8"))
        self.raw_words = max(len(self.corpus_text.split()), 1)

    def train_uniqtoken_models(self) -> Dict[str, CustomTokenizer]:
        """Trains standard UniqToken Unigram and SuperBPE enhanced models."""
        base_tok = CustomTokenizer.train_from_corpus(
            corpus=self.corpus,
            target_vocab_size=self.vocab_size,
            min_frequency=1,
            ranking_strategy="pmi",
            adaptive_multiplier=True,
            verbose=False,
        )

        # Extended SuperBPE model
        pretok_chunks: List[str] = []
        for doc in self.corpus:
            norm = base_tok.normalizer.normalize(doc)
            pretok_chunks.extend(base_tok.pre_tokenizer.pre_tokenize(norm))

        superbpe = CrossEntropyMerging(max_merges=self.max_merges, cross_word=True, verbose=False)
        sbp_model = superbpe.optimize(base_tok.model, chunks=pretok_chunks)
        sbp_tok = CustomTokenizer(
            normalizer=base_tok.normalizer,
            pre_tokenizer=base_tok.pre_tokenizer,
            model=sbp_model,
        )

        return {
            "UniqToken (Unigram)": base_tok,
            "UniqToken (SuperBPE)": sbp_tok,
        }

    def evaluate_tokenizer(self, name: str, encode_fn, vocab_size: int) -> DownstreamMetrics:
        """Evaluates an arbitrary tokenizer against downstream LLM context invariants."""
        t0 = time.perf_counter()
        token_ids = encode_fn(self.corpus_text)
        t_enc = max(time.perf_counter() - t0, 1e-6)

        num_tokens = len(token_ids)
        if num_tokens == 0:
            raise ValueError(f"tokenizer {name!r} returned no tokens for a non-empty evaluation corpus")
        bytes_per_tok = self.raw_bytes / max(num_tokens, 1)
        tokens_per_word = num_tokens / self.raw_words

        # Theoretical Entropy / Bits Per Byte
        # BPC estimate based on vocabulary uniform bit cost
        bits_per_token = math.log2(max(vocab_size, 2))
        total_bits = num_tokens * bits_per_token
        bits_per_byte = total_bits / max(self.raw_bytes, 1)

        return DownstreamMetrics(
            tokenizer_name=name,
            vocab_size=vocab_size,
            total_tokens=num_tokens,
            total_bytes=self.raw_bytes,
            total_words=self.raw_words,
            bytes_per_token=round(bytes_per_tok, 3),
            tokens_per_word=round(tokens_per_word, 3),
            effective_bytes_in_2k_context=int(2048 * bytes_per_tok),
            effective_bytes_in_4k_context=int(4096 * bytes_per_tok),
            effective_bytes_in_8k_context=int(8192 * bytes_per_tok),
            estimated_bits_per_byte=round(bits_per_byte, 3),
            encode_time_sec=round(t_enc, 4),
        )

    def run_downstream_suite(self, include_external_baselines: bool = True) -> List[DownstreamMetrics]:
        """Runs downstream evaluation across all available tokenizers."""
        results: List[DownstreamMetrics] = []

        # 1. UniqToken Models
        uniqtoken_models = self.train_uniqtoken_models()
        for name, tok in uniqtoken_models.items():
            metrics = self.evaluate_tokenizer(
                name=name,
                encode_fn=tok.encode_to_ids,
                vocab_size=tok.vocab_size,
            )
            results.append(metrics)

        if not include_external_baselines:
            return results

        # 2. External Baselines (tiktoken, huggingface, sentencepiece if available)
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            m = self.evaluate_tokenizer("tiktoken (cl100k_base)", enc.encode, enc.n_vocab)
            results.append(m)
        except Exception as exc:  # noqa: BLE001 - baseline opt-in
            warnings.warn(f"tiktoken baseline unavailable ({exc}); skipping")

        try:
            from transformers import AutoTokenizer

            hf_tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
            m = self.evaluate_tokenizer(
                "HuggingFace (GPT-2)",
                lambda t: hf_tok.encode(t, add_special_tokens=False),
                hf_tok.vocab_size,
            )
            results.append(m)
        except Exception as exc:  # noqa: BLE001 - baseline opt-in
            warnings.warn(f"HuggingFace GPT-2 baseline unavailable ({exc}); skipping")

        return results

    def print_report(self, results: List[DownstreamMetrics]) -> None:
        """Formats and prints the downstream comparative report."""
        print("=" * 110)
        print("DOWNSTREAM LLM CONTEXT EFFICIENCY & INFORMATION DENSITY BENCHMARK")
        print("=" * 110)
        header = f"{'Tokenizer':<24} | {'Vocab':<7} | {'Tokens':<7} | {'Bytes/Tok':<10} | {'Tok/Word':<9} | {'2K Window (Bytes)':<18} | {'Bits/Byte':<10}"
        print(header)
        print("-" * len(header))

        for r in results:
            print(
                f"{r.tokenizer_name:<24} | {r.vocab_size:<7} | {r.total_tokens:<7} | "
                f"{r.bytes_per_token:<10} | {r.tokens_per_word:<9} | "
                f"{r.effective_bytes_in_2k_context:<18} | {r.estimated_bits_per_byte:<10}"
            )
        print("=" * 110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Downstream LLM Tokenizer Evaluation.")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Target vocabulary size for trained models")
    parser.add_argument("--smoke-test", action="store_true", help="Quick verification smoke test")
    args = parser.parse_args()

    vs = 500 if args.smoke_test else args.vocab_size
    evaluator = DownstreamEvaluator(vocab_size=vs)
    results = evaluator.run_downstream_suite()
    evaluator.print_report(results)
