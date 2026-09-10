"""
Held-Out Tokenizer Measurement Harness.

Evaluates held-out tokenizer compression and context-density proxies:
1. Context Window Information Density (Effective Bytes per Context Window).
2. Bytes per token.
3. Tokens per Unicode code point across linguistic domains.

This module does not train or evaluate a language model.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uniqtoken.tokenizer import CustomTokenizer
from benchmarks.train_toy_transformer import (
    PRETRAINING_CORPUS,
    _require_vocab_budget,
    _split_documents,
    train_superbpe_tokenizer,
)


@dataclass
class DownstreamMetrics:
    tokenizer_name: str
    model_kind: str
    vocab_size: int
    total_tokens: int
    total_bytes: int
    total_words: int
    bytes_per_token: float
    tokens_per_unicode_character: float
    effective_bytes_in_2k_context: int
    effective_bytes_in_4k_context: int
    effective_bytes_in_8k_context: int
    encode_time_sec: float


class DownstreamEvaluator:
    """Measure tokenizer-only properties on documents excluded from training."""

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
        training_corpus: Optional[List[str]] = None,
        evaluation_corpus: Optional[List[str]] = None,
    ):
        if corpus is not None and (training_corpus is not None or evaluation_corpus is not None):
            raise ValueError("pass either corpus to split or explicit training_corpus/evaluation_corpus, not both")
        if (training_corpus is None) != (evaluation_corpus is None):
            raise ValueError("training_corpus and evaluation_corpus must be provided together")
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if max_merges < 1:
            raise ValueError("max_merges must be positive")

        self.vocab_size = vocab_size
        self.max_merges = max_merges
        if training_corpus is not None and evaluation_corpus is not None:
            self.training_corpus = list(training_corpus)
            self.evaluation_corpus = list(evaluation_corpus)
        else:
            source_corpus = list(corpus if corpus is not None else PRETRAINING_CORPUS)
            self.training_corpus, _, self.evaluation_corpus = _split_documents(source_corpus)

        if not self.training_corpus or not self.evaluation_corpus:
            raise ValueError("benchmark requires non-empty training and evaluation corpora")
        overlap = set(self.training_corpus).intersection(self.evaluation_corpus)
        if overlap:
            raise ValueError("training and evaluation corpora overlap by document identity")

        self.corpus = self.evaluation_corpus
        self.corpus_text = "\n\n".join(self.evaluation_corpus)
        self.raw_bytes = len(self.corpus_text.encode("utf-8"))
        self.raw_words = max(len(self.corpus_text.split()), 1)

    def train_uniqtoken_models(self) -> Dict[str, CustomTokenizer]:
        """Trains standard UniqToken Unigram and SuperBPE enhanced models."""
        base_tok = CustomTokenizer.train_from_corpus(
            corpus=self.training_corpus,
            target_vocab_size=self.vocab_size,
            min_edge_log_prob=float("-inf"),
            min_frequency=1,
            ranking_strategy="pmi",
            adaptive_multiplier=True,
            verbose=False,
        )
        _require_vocab_budget(base_tok, self.vocab_size, "UniqToken Unigram")

        sbp_tok = train_superbpe_tokenizer(self.training_corpus, self.vocab_size, self.max_merges)

        return {
            "UniqToken (Unigram)": base_tok,
            "UniqToken (SuperBPE)": sbp_tok,
        }

    def evaluate_tokenizer(self, name: str, encode_fn, vocab_size: int) -> DownstreamMetrics:
        """Evaluates an arbitrary tokenizer on the held-out tokenizer corpus."""
        t0 = time.perf_counter()
        token_ids = encode_fn(self.corpus_text)
        t_enc = max(time.perf_counter() - t0, 1e-6)

        num_tokens = len(token_ids)
        if num_tokens == 0:
            raise ValueError(f"tokenizer {name!r} returned no tokens for a non-empty evaluation corpus")
        bytes_per_tok = self.raw_bytes / max(num_tokens, 1)
        tokens_per_unicode_character = num_tokens / max(len(self.corpus_text), 1)

        return DownstreamMetrics(
            tokenizer_name=name,
            model_kind="tokenizer_only",
            vocab_size=vocab_size,
            total_tokens=num_tokens,
            total_bytes=self.raw_bytes,
            total_words=self.raw_words,
            bytes_per_token=round(bytes_per_tok, 3),
            tokens_per_unicode_character=round(tokens_per_unicode_character, 3),
            effective_bytes_in_2k_context=int(2048 * bytes_per_tok),
            effective_bytes_in_4k_context=int(4096 * bytes_per_tok),
            effective_bytes_in_8k_context=int(8192 * bytes_per_tok),
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

    def evaluate_low_resource_languages(
        self,
        corpora: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates compression, tokens per Unicode code point, and fallback rates across
        language corpora (Swahili, Yoruba, Malayalam, Amharic) against external baselines.

        Args:
            corpora: Optional mapping from language dataset name to text. If None,
                loads the underrepresented corpora directly from TokenizerBenchmarkSuite.

        Returns:
            List of dictionaries containing evaluation metrics for each language.
        """
        if corpora is None:
            from benchmarks.benchmark_suite import TokenizerBenchmarkSuite

            corpora = {
                k: v
                for k, v in TokenizerBenchmarkSuite.BENCHMARK_CORPORA.items()
                if k in ("Agglutinative_Swahili", "Tonal_Yoruba", "Agglutinative_Malayalam", "Geez_Amharic")
            }

        suite_models = self.train_uniqtoken_models()
        tok = suite_models.get("UniqToken (SuperBPE)") or suite_models["UniqToken (Unigram)"]

        overlap = set(self.training_corpus).intersection(corpora.values())
        if overlap:
            raise ValueError("low-resource evaluation corpus overlaps tokenizer training data")

        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = None

        results: List[Dict[str, Any]] = []
        for name, text in corpora.items():
            raw_bytes = len(text.encode("utf-8"))
            characters = max(len(text), 1)

            tokens_with_offsets = tok.encode_with_offsets(text)
            tok_count = len(tokens_with_offsets)
            bpt = round(raw_bytes / max(tok_count, 1), 2)
            fertility = round(tok_count / characters, 2)

            fb_count = sum(1 for t in tokens_with_offsets if t.text.startswith("<0x") and t.text.endswith(">"))
            fallback_pct = round((fb_count / max(tok_count, 1)) * 100.0, 2)

            if enc is not None:
                tt_tokens = len(enc.encode(text))
                tt_bpt = round(raw_bytes / max(tt_tokens, 1), 2)
                tt_fert = round(tt_tokens / characters, 2)
            else:
                tt_tokens = 0
                tt_bpt = 0.0
                tt_fert = 0.0

            results.append(
                {
                    "dataset": name,
                    "raw_bytes": raw_bytes,
                    "tokens": tok_count,
                    "bytes_per_token": bpt,
                    "tokens_per_unicode_character": fertility,
                    "fallback_rate_pct": fallback_pct,
                    "tiktoken_tokens": tt_tokens,
                    "tiktoken_bytes_per_token": tt_bpt,
                    "tiktoken_tokens_per_unicode_character": tt_fert,
                }
            )
        return results

    def print_report(self, results: List[DownstreamMetrics]) -> None:
        """Formats and prints the downstream comparative report."""
        print("=" * 110)
        print("HELD-OUT TOKENIZER CONTEXT-DENSITY BENCHMARK")
        print("=" * 110)
        header = f"{'Tokenizer':<24} | {'Vocab':<7} | {'Tokens':<7} | {'Bytes/Tok':<10} | {'Tok/char':<9} | {'2K Window (Bytes)':<18}"
        print(header)
        print("-" * len(header))

        for r in results:
            print(
                f"{r.tokenizer_name:<24} | {r.vocab_size:<7} | {r.total_tokens:<7} | "
                f"{r.bytes_per_token:<10} | {r.tokens_per_unicode_character:<9} | "
                f"{r.effective_bytes_in_2k_context:<18}"
            )
        print("=" * 110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run held-out tokenizer-only measurements.")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Target vocabulary size for trained models")
    parser.add_argument("--smoke-test", action="store_true", help="Quick verification smoke test")
    args = parser.parse_args()

    vs = 500 if args.smoke_test else args.vocab_size
    evaluator = DownstreamEvaluator(vocab_size=vs)
    results = evaluator.run_downstream_suite()
    evaluator.print_report(results)
