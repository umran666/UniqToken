from __future__ import annotations

import math
import os
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .seed_builder import SeedToken, SeedVocabularyBuilder
from .streaming_counter import StreamingChunkCounter
from .trie import PrefixTrie
from .unigram_lattice import UnigramLattice
from .byte_codec import ByteFallbackEngine

try:
    import uniqtoken_core

    _HAS_UNIQTOKEN_CORE = hasattr(uniqtoken_core, "RustPrefixTrie")
except ImportError:
    uniqtoken_core = None  # type: ignore[assignment]
    _HAS_UNIQTOKEN_CORE = False


@dataclass
class UnigramModel:
    """
    Trained Unigram Tokenizer Model.
    """

    vocab: Dict[str, float]  # token -> log_probability
    token_to_id: Dict[str, int]  # token -> integer ID
    id_to_token: Dict[int, str]  # integer ID -> token
    special_tokens: List[str]
    max_subword_len: int = 16
    byte_fallback: bool = True
    unk_token: str = "<|unk|>"

    @property
    def vocab_size(self) -> int:
        # IDs may contain holes when importing SentencePiece or HF models;
        # embedding tables must cover the highest assigned ID.
        return max(self.token_to_id.values(), default=-1) + 1

    def _cache_signature(self) -> Tuple[int, int]:
        # ponytail: assumes vocab dict replaced not mutated in-place; add version counter if external mutation needed
        return (id(self.vocab), len(self.vocab))

    def _sync_cache(self) -> None:
        """Invalidates caches if the vocab object or size changed since they were built."""
        sig = self._cache_signature()
        if self.__dict__.get("_cache_sig") != sig:
            self.clear_cache()
            self._cache_sig = sig

    def _get_trie(self) -> PrefixTrie:
        """Builds (and caches) a PrefixTrie over the current vocab for fast lattice search."""
        self._sync_cache()
        trie = self.__dict__.get("_trie")
        if trie is None:
            trie = PrefixTrie.from_vocab(self.vocab, self.token_to_id)
            self._trie = trie
        return trie

    def _get_rust_trie(self) -> Optional[Any]:
        """Builds (and caches) a native RustPrefixTrie if uniqtoken_core is available."""
        if not _HAS_UNIQTOKEN_CORE:
            return None
        self._sync_cache()
        rust_trie = self.__dict__.get("_rust_trie")
        if rust_trie is None:
            rust_trie = uniqtoken_core.RustPrefixTrie(self.max_subword_len)
            for token, log_p in self.vocab.items():
                tid = self.token_to_id.get(token)
                if tid is None:
                    raise ValueError(f"vocabulary token {token!r} has no integer ID")
                rust_trie.insert(token, log_p, tid)
            self._rust_trie = rust_trie
        return rust_trie

    _FAST_PATH_MAX_LEN = 6
    _MAX_CACHE_SIZE = 10000

    def clear_cache(self) -> None:
        """Clears cached PrefixTrie and segmentation memoization tables."""
        if "_trie" in self.__dict__:
            del self.__dict__["_trie"]
        if "_rust_trie" in self.__dict__:
            del self.__dict__["_rust_trie"]
        if "_seg_cache" in self.__dict__:
            del self.__dict__["_seg_cache"]
        if "_cache_sig" in self.__dict__:
            del self.__dict__["_cache_sig"]

    def _get_seg_cache(self) -> Dict[str, List[Tuple[str, int, int]]]:
        self._sync_cache()
        cache = self.__dict__.get("_seg_cache")
        if cache is None:
            cache = {}
            self._seg_cache = cache
        return cache

    def _encode_fast(self, text: str) -> Optional[List[Tuple[str, int, int]]]:
        """
        Exact Viterbi 1-best for short chunks without constructing a lattice object.

        Reproduces UnigramLattice edge semantics (including byte-fallback
        only-when-no-vocab-edge and longest-edge-first tie-breaking). Returns
        (token, start, end) triples, or None to defer to the full lattice.
        """
        length = len(text)
        if length < 2 or length > self._FAST_PATH_MAX_LEN:
            return None

        max_len = min(self.max_subword_len, length)
        neg_inf = float("-inf")
        dp = [neg_inf] * (length + 1)
        back: List[Optional[Tuple[int, List[str]]]] = [None] * (length + 1)
        dp[0] = 0.0

        for i in range(1, length + 1):
            best_score = neg_inf
            best_edge: Optional[Tuple[int, List[str]]] = None
            start = max(0, i - max_len)
            for j in range(start, i):
                token = text[j:i]
                log_p = self.vocab.get(token)
                if log_p is not None and dp[j] > neg_inf:
                    score = dp[j] + log_p
                    if score > best_score:
                        best_score = score
                        best_edge = (j, [token])

            # Byte fallback edge from (i - 1) to i applies if no vocab edge starts at (i - 1)
            if self.byte_fallback and dp[i - 1] > neg_inf:
                has_vocab_edge = False
                for end in range(i, min(length + 1, i - 1 + max_len + 1)):
                    if text[i - 1 : end] in self.vocab:
                        has_vocab_edge = True
                        break
                if not has_vocab_edge:
                    char = text[i - 1]
                    byte_tokens = ByteFallbackEngine.char_to_byte_tokens(char)
                    log_p = sum(self.vocab.get(b, -UnigramLattice.DEFAULT_BYTE_PENALTY) for b in byte_tokens)
                    score = dp[i - 1] + log_p
                    if score > best_score:
                        best_score = score
                        best_edge = (i - 1, byte_tokens)

            if best_edge is None:
                return None  # disconnected; defer to the full lattice

            dp[i] = best_score
            back[i] = best_edge

        result: List[Tuple[str, int, int]] = []
        i = length
        while i > 0:
            edge = back[i]
            assert edge is not None
            j, tokens = edge
            for token in reversed(tokens):
                result.append((token, j, i))
            i = j
        result.reverse()
        return result

    def encode(self, text: str) -> List[str]:
        if len(text) == 1 and text in self.vocab:
            return [text]
        spans = self.encode_with_spans(text)
        return [token for token, _, _ in spans]

    def encode_batch(self, chunks: List[str]) -> List[List[str]]:
        """Batched encode — single trie/cache lookup, one Rust batch if available."""
        if not chunks:
            return []
        # ponytail: batch cuts 4600→100 FFI/cache checks; Rust batch when compiled else hoisted Python
        rust_trie = self._get_rust_trie()
        if rust_trie is not None:
            # uniqtoken_core is the module-level import (or None);
            # do NOT re-import it here — a stale site-packages
            # module would reject this module's RustPrefixTrie instances.
            core = uniqtoken_core
            if core is not None:
                try:
                    # ponytail: rust_encode_tokens_batch returns Vec<Vec<String>> directly — no ViterbiSpan wrapper, single FFI
                    if hasattr(core, "rust_encode_tokens_batch"):
                        # ponytail: no span cache for token-only batch; encode_with_spans will compute exact spans on miss
                        return core.rust_encode_tokens_batch(chunks, rust_trie, self.byte_fallback)
                    batch_spans = core.rust_viterbi_decode_batch(chunks, rust_trie, self.byte_fallback)
                    res: List[List[str]] = []
                    cache = self._get_seg_cache()
                    for chunk, rust_span_list in zip(chunks, batch_spans):
                        lst = [s.token for s in rust_span_list]
                        if len(chunk) <= 64 and len(cache) < self._MAX_CACHE_SIZE and chunk not in cache:
                            cache[chunk] = [(s.token, s.start, s.end) for s in rust_span_list]
                        res.append(lst)
                    return res
                except (ImportError, AttributeError, ValueError):
                    pass
        # Python fallback — hoisted trie/cache
        cache = self._get_seg_cache()
        trie = self._get_trie()
        out: List[List[str]] = []
        for chunk in chunks:
            if len(chunk) == 1 and chunk in self.vocab:
                out.append([chunk])
                continue
            cached = cache.get(chunk)
            if cached is not None:
                out.append([t for t, _, _ in cached])
                continue
            fast = self._encode_fast(chunk)
            chunk_spans: List[Tuple[str, int, int]]
            if fast is not None:
                chunk_spans = fast
            else:
                lattice = UnigramLattice(
                    chunk,
                    self.vocab,
                    max_subword_len=self.max_subword_len,
                    byte_fallback=self.byte_fallback,
                    unk_token=self.unk_token,
                    trie=trie,
                )
                edges, _ = lattice.viterbi_edges()
                chunk_spans = [(token, edge.start, edge.end) for edge in edges for token in edge.tokens]
            if len(chunk) <= 64 and len(cache) < self._MAX_CACHE_SIZE:
                cache[chunk] = chunk_spans
            out.append([t for t, _, _ in chunk_spans])
        return out

    def encode_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Encode text and retain normalized character spans for every output token."""
        if len(text) == 1 and text in self.vocab:
            return [(text, 0, 1)]

        cache = self._get_seg_cache()
        cached = cache.get(text)
        if cached is not None:
            return list(cached)

        # 1. Native Rust Viterbi engine dispatch (if uniqtoken_core compiled)
        rust_trie = self._get_rust_trie()
        if rust_trie is not None:
            try:
                rust_spans = uniqtoken_core.rust_viterbi_decode(
                    text,
                    rust_trie,
                    self.byte_fallback,
                )
                spans = [(s.token, s.start, s.end) for s in rust_spans]
                if len(text) <= 64 and len(cache) < self._MAX_CACHE_SIZE:
                    cache[text] = spans
                return spans
            except (ImportError, AttributeError, ValueError):
                pass

        # 2. Pure Python fast path or full lattice DAG
        fast = self._encode_fast(text)
        if fast is not None:
            spans = fast
        else:
            lattice = UnigramLattice(
                text,
                self.vocab,
                max_subword_len=self.max_subword_len,
                byte_fallback=self.byte_fallback,
                unk_token=self.unk_token,
                trie=self._get_trie(),
            )
            edges, _ = lattice.viterbi_edges()
            spans = [(token, edge.start, edge.end) for edge in edges for token in edge.tokens]

        if len(text) <= 64 and len(cache) < self._MAX_CACHE_SIZE:
            cache[text] = spans
        return spans

    def sample(self, text: str, alpha: float = 0.5) -> List[str]:
        if len(text) == 1 and text in self.vocab:
            return [text]
        lattice = UnigramLattice(
            text,
            self.vocab,
            max_subword_len=self.max_subword_len,
            byte_fallback=self.byte_fallback,
            unk_token=self.unk_token,
            trie=self._get_trie(),
        )
        return lattice.sample(alpha=alpha)

    def encode_to_ids(self, text: str) -> List[int]:
        tokens = self.encode(text)
        unk_id = self.token_to_id.get(self.unk_token, 0)
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def sample_to_ids(self, text: str, alpha: float = 0.5) -> List[int]:
        tokens = self.sample(text, alpha=alpha)
        unk_id = self.token_to_id.get(self.unk_token, 0)
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def decode(self, token_ids: List[int], space_char: str = "\u2581") -> str:
        from .byte_codec import ByteFallbackEngine

        tokens = [self.id_to_token.get(i, self.unk_token) for i in token_ids]
        return ByteFallbackEngine.decode_tokens(tokens, space_char=space_char)


class UnigramTrainer:
    """
    Expectation-Maximization (EM) & Iterative Likelihood Pruning Trainer for Unigram Tokenizer.
    """

    def __init__(
        self,
        target_vocab_size: int = 8000,
        seed_multiplier: float = 3.0,
        max_ngram_length: int = 16,
        min_frequency: int = 2,
        byte_fallback: bool = True,
        prune_rate: float = 0.75,
        em_sub_iterations: int = 2,
        special_tokens: List[str] | None = None,
        ranking_strategy: str = "char_savings",
        adaptive_multiplier: bool = False,
        max_edges_per_node: Optional[int] = None,
        min_edge_log_prob: Optional[float] = None,
        convergence_tolerance: float = 1e-4,
        script_balance_temperature: Optional[float] = None,
        min_boundary_entropy: Optional[float] = None,
        length_exponent: float = 1.0,
        pruning_length_exponent: float = 0.0,
        show_progress: bool = True,
        streaming: bool = False,
        chunk_size_bytes: int = 500 * 1024 * 1024,
    ):
        """Initializes the Unigram EM and Iterative Likelihood Pruning Trainer.

        Args:
            target_vocab_size: Target final vocabulary size.
            seed_multiplier: Multiplier for seed candidate vocabulary pool size.
            max_ngram_length: Maximum character length for candidate subwords.
            min_frequency: Minimum occurrence count for seed token candidates.
            byte_fallback: Enable fallback to raw byte tokens for unknown characters.
            prune_rate: Proportion of excess vocabulary to eliminate per EM round.
            em_sub_iterations: Number of E-step / M-step sub-iterations per round.
            special_tokens: Optional list of special token strings.
            ranking_strategy: Scoring heuristic for seed vocabulary candidate ranking.
            adaptive_multiplier: Adapt seed pool multiplier dynamically by corpus entropy.
            max_edges_per_node: Lattice edge pruning threshold.
            min_edge_log_prob: Minimum log probability threshold for lattice edges.
            convergence_tolerance: Delta log-likelihood threshold for EM convergence.
            script_balance_temperature: Temperature for script-balanced quota allocation.
            min_boundary_entropy: Minimum branch entropy threshold for seed candidates.
            length_exponent: Length exponent for seed vocabulary scoring regularization.
            pruning_length_exponent: Length regularization exponent for pruning candidate scoring.
            show_progress: Display real-time progress indicators during training.
        """
        if target_vocab_size <= 0:
            raise ValueError("target_vocab_size must be greater than zero")
        if seed_multiplier <= 0:
            raise ValueError("seed_multiplier must be greater than zero")
        if max_ngram_length < 1:
            raise ValueError("max_ngram_length must be at least one")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least one")
        if not 0 < prune_rate <= 1:
            raise ValueError("prune_rate must be in the interval (0, 1]")
        if em_sub_iterations < 1:
            raise ValueError("em_sub_iterations must be at least one")
        self.target_vocab_size = target_vocab_size
        self.seed_multiplier = seed_multiplier
        self.max_ngram_length = max_ngram_length
        self.min_frequency = min_frequency
        self.byte_fallback = byte_fallback
        self.prune_rate = prune_rate
        self.em_sub_iterations = em_sub_iterations
        self.special_tokens = special_tokens
        self.ranking_strategy = ranking_strategy
        self.adaptive_multiplier = adaptive_multiplier
        self.max_edges_per_node = max_edges_per_node
        self.min_edge_log_prob = min_edge_log_prob
        self.convergence_tolerance = convergence_tolerance
        self.script_balance_temperature = script_balance_temperature
        self.min_boundary_entropy = min_boundary_entropy
        self.length_exponent = length_exponent
        self.pruning_length_exponent = pruning_length_exponent
        self.show_progress = show_progress
        self.streaming = streaming
        self.chunk_size_bytes = chunk_size_bytes

    def train(
        self,
        pre_tokenized_chunks: Iterable[str] | Mapping[str, int],
        verbose: bool = True,
        show_progress: Optional[bool] = None,
        streaming: Optional[bool] = None,
        chunk_size_bytes: Optional[int] = None,
    ) -> UnigramModel:
        """Runs the full EM training and pruning loop with convergence checks and beam pruning.

        Args:
            pre_tokenized_chunks: Stream or collection of pre-tokenized text segments.
            verbose: Print detailed per-round status logs.
            show_progress: Display dynamic real-time progress indicators.
                If None, defaults to the trainer's show_progress configuration.

        Returns:
            Trained UnigramModel containing vocabulary, token IDs, and special tokens.
        """
        if show_progress is None:
            show_progress = self.show_progress
        if os.environ.get("UNIQTOKEN_NO_PROGRESS", "0") == "1":
            show_progress = False
        if tqdm is None:
            show_progress = False
        if streaming is None:
            streaming = self.streaming
        if chunk_size_bytes is None:
            chunk_size_bytes = self.chunk_size_bytes

        # Step 1: Pre-aggregate chunk frequencies (in-memory Counter or disk-backed StreamingChunkCounter)
        created_counter: Optional[StreamingChunkCounter] = None
        if isinstance(pre_tokenized_chunks, Mapping):
            chunk_counts = pre_tokenized_chunks
            finalize_fn = getattr(chunk_counts, "finalize", None)
            if callable(finalize_fn):
                finalize_fn()
        elif streaming:
            created_counter = StreamingChunkCounter(chunk_size_bytes=chunk_size_bytes)
            created_counter.update(pre_tokenized_chunks)
            created_counter.finalize()
            chunk_counts = created_counter
        else:
            chunk_counts = Counter(pre_tokenized_chunks)

        try:
            return self._train_with_counts(
                chunk_counts=chunk_counts,
                verbose=verbose,
                show_progress=show_progress,
            )
        finally:
            if created_counter is not None:
                created_counter.close()

    def _train_with_counts(
        self,
        chunk_counts: Mapping[str, int],
        verbose: bool,
        show_progress: bool,
    ) -> UnigramModel:

        # Step 2: Build Seed Vocabulary
        seed_builder = SeedVocabularyBuilder(
            target_vocab_size=self.target_vocab_size,
            seed_multiplier=self.seed_multiplier,
            max_ngram_length=self.max_ngram_length,
            min_frequency=self.min_frequency,
            byte_fallback=self.byte_fallback,
            special_tokens=self.special_tokens,
            ranking_strategy=self.ranking_strategy,
            adaptive_multiplier=self.adaptive_multiplier,
            script_balance_temperature=self.script_balance_temperature,
            min_boundary_entropy=self.min_boundary_entropy,
            length_exponent=self.length_exponent,
        )

        seed_tokens: List[SeedToken] = seed_builder.build_seed_vocab(chunk_counts)
        required_tokens: Set[str] = {t.token for t in seed_tokens if t.is_required}

        # Step 3: Initialize log probabilities from seed counts
        total_seed_freq = sum(t.frequency for t in seed_tokens)
        current_vocab_log_probs: Dict[str, float] = {
            t.token: math.log(max(t.frequency, 1) / total_seed_freq) for t in seed_tokens
        }

        total_to_prune = max(1, len(current_vocab_log_probs) - self.target_vocab_size)
        pbar = None
        total_corpus_bytes = 0
        if show_progress:
            total_corpus_bytes = sum(
                len(c.encode("utf-8", errors="replace")) * count for c, count in chunk_counts.items()
            )
            pbar = tqdm(
                total=total_to_prune,
                desc="EM Training & Pruning",
                unit="tok",
                dynamic_ncols=True,
                leave=True,
            )

        def _log(msg: str) -> None:
            """Log progress message using tqdm.write to avoid interrupting progress bar."""
            if pbar is not None and hasattr(tqdm, "write"):
                tqdm.write(msg)
            else:
                print(msg)

        if verbose:
            _log(
                f"[EM Trainer] Seed Vocab Size: {len(current_vocab_log_probs)} (Target: {self.target_vocab_size}, Required: {len(required_tokens)})"
            )

        round_num = 1
        delta_str = "init"
        em_throughput = 0.0

        try:
            # Step 4: Iterative EM Optimization & Pruning Loop
            while len(current_vocab_log_probs) > self.target_vocab_size:
                num_candidates = sum(1 for tok in current_vocab_log_probs if tok not in required_tokens)
                if pbar is not None:
                    pbar.set_postfix(
                        {
                            "vocab": len(current_vocab_log_probs),
                            "ΔL": delta_str,
                            "candidates": num_candidates,
                            "throughput": f"{em_throughput:.2f} MB/s",
                            "round": round_num,
                        },
                        refresh=False,
                    )

                # --- E-STEP & M-STEP SUB-ITERATIONS ---
                prev_log_lik = -float("inf")
                round_em_start = time.perf_counter()
                for sub_iter in range(self.em_sub_iterations):
                    expected_counts: Dict[str, float] = {}
                    total_corpus_log_lik = 0.0
                    can_use_rust_em = (
                        _HAS_UNIQTOKEN_CORE
                        and self.min_edge_log_prob is None
                        and self.max_edges_per_node is None
                        and self.max_ngram_length >= 16
                    )
                    if can_use_rust_em:
                        try:
                            rust_trie = uniqtoken_core.RustPrefixTrie(self.max_ngram_length)
                            for t, lp in current_vocab_log_probs.items():
                                rust_trie.insert(t, lp, 0)
                            for chunk, count in chunk_counts.items():
                                if chunk in required_tokens and (chunk.startswith("<|") and chunk.endswith("|>")):
                                    expected_counts[chunk] = expected_counts.get(chunk, 0.0) + count
                                    continue
                                chunk_exp, chunk_log_lik = uniqtoken_core.rust_forward_backward_expectations(
                                    chunk, rust_trie, 1.0
                                )
                                total_corpus_log_lik += chunk_log_lik * count
                                for tok, exp_val in chunk_exp.items():
                                    expected_counts[tok] = expected_counts.get(tok, 0.0) + (exp_val * count)
                        except Exception:
                            can_use_rust_em = False
                            expected_counts.clear()
                            total_corpus_log_lik = 0.0

                    if not can_use_rust_em:
                        trie = PrefixTrie.from_vocab(current_vocab_log_probs)
                        for chunk, count in chunk_counts.items():
                            if chunk in required_tokens and (chunk.startswith("<|") and chunk.endswith("|>")):
                                expected_counts[chunk] = expected_counts.get(chunk, 0.0) + count
                                continue

                            lattice = UnigramLattice(
                                text=chunk,
                                vocab_log_probs=current_vocab_log_probs,
                                max_subword_len=self.max_ngram_length,
                                byte_fallback=self.byte_fallback,
                                trie=trie,
                                max_edges_per_node=self.max_edges_per_node,
                                min_edge_log_prob=self.min_edge_log_prob,
                            )

                            chunk_exp, chunk_log_lik = lattice.forward_backward()
                            total_corpus_log_lik += chunk_log_lik * count

                            for tok, exp_val in chunk_exp.items():
                                expected_counts[tok] = expected_counts.get(tok, 0.0) + (exp_val * count)

                    total_expected = sum(expected_counts.values())
                    if total_expected <= 0:
                        break

                    delta_log_lik = abs(total_corpus_log_lik - prev_log_lik)
                    prev_log_lik = total_corpus_log_lik
                    em_elapsed = time.perf_counter() - round_em_start
                    bytes_processed = total_corpus_bytes * (sub_iter + 1)
                    em_throughput = (bytes_processed / (1024 * 1024)) / em_elapsed if em_elapsed > 0 else 0.0

                    if math.isinf(delta_log_lik):
                        delta_str = "init"
                    elif delta_log_lik >= 1.0:
                        delta_str = f"{delta_log_lik:.4f}"
                    else:
                        delta_str = f"{delta_log_lik:.4e}"

                    if pbar is not None:
                        pbar.set_postfix(
                            {
                                "vocab": len(current_vocab_log_probs),
                                "ΔL": delta_str,
                                "candidates": num_candidates,
                                "throughput": f"{em_throughput:.2f} MB/s",
                                "round": round_num,
                            },
                            refresh=False,
                        )
                        pbar.refresh()

                    current_vocab_log_probs = {
                        tok: math.log(max(expected_counts.get(tok, 1e-12) / total_expected, 1e-12))
                        for tok in current_vocab_log_probs
                    }

                    if sub_iter > 0 and delta_log_lik < self.convergence_tolerance:
                        break

                # --- PRUNING STEP ---
                if not expected_counts or total_expected <= 0:
                    if verbose:
                        _log("  No expectations computed; terminating EM pruning.")
                    break

                current_size = len(current_vocab_log_probs)
                if current_size <= self.target_vocab_size:
                    break

                excess = current_size - self.target_vocab_size
                if excess <= 500 or round_num >= 3:
                    num_to_prune = excess
                else:
                    num_to_prune = max(1, int(excess * self.prune_rate))
                # Candidates are scored by their contribution to corpus likelihood with length regularization beta
                # Score(t) = E[count(t)] * log p(t) * (byte_len ^ beta).
                # When beta > 0, longer subwords receive higher survival pressure (more negative score).
                candidate_scores: List[Tuple[str, float]] = []

                for tok, log_p in current_vocab_log_probs.items():
                    if tok in required_tokens:
                        continue  # Required tokens are immune to pruning
                    exp_c = expected_counts.get(tok, 0.0)
                    if self.pruning_length_exponent > 0.0:
                        byte_len = max(len(tok.encode("utf-8")), 1)
                        score = exp_c * log_p * (float(byte_len) ** self.pruning_length_exponent)
                    else:
                        score = exp_c * log_p
                    candidate_scores.append((tok, score))

                pruning_candidate_count = len(candidate_scores)

                # Deterministic: remove the least negative scores before valuable
                # high-count tokens whose contribution is strongly negative.
                candidate_scores.sort(key=lambda x: (-x[1], x[0]))

                # Prune lowest candidates
                tokens_to_remove = set(tok for tok, _ in candidate_scores[:num_to_prune])

                new_vocab_log_probs: Dict[str, float] = {
                    tok: log_p for tok, log_p in current_vocab_log_probs.items() if tok not in tokens_to_remove
                }

                current_vocab_log_probs = new_vocab_log_probs
                new_candidate_count = sum(1 for tok in current_vocab_log_probs if tok not in required_tokens)

                if pbar is not None:
                    pbar.update(len(tokens_to_remove))
                    pbar.set_postfix(
                        {
                            "vocab": len(current_vocab_log_probs),
                            "ΔL": delta_str,
                            "candidates": new_candidate_count,
                            "throughput": f"{em_throughput:.2f} MB/s",
                            "round": round_num,
                        }
                    )

                if verbose:
                    _log(
                        f"[EM Round {round_num:>2}] Vocab: {len(current_vocab_log_probs):>5} | Pruned: {len(tokens_to_remove):>4} | Candidates: {pruning_candidate_count:>5} | ΔL: {delta_str} | Corpus LogLik: {total_corpus_log_lik:.2f}"
                    )

                round_num += 1
        finally:
            if pbar is not None:
                pbar.close()

        # Step 5: Final Probability Re-normalization
        total_p = sum(math.exp(log_p) for log_p in current_vocab_log_probs.values())
        final_vocab = {tok: math.log(math.exp(log_p) / total_p) for tok, log_p in current_vocab_log_probs.items()}

        # Step 6: Build Integer Token IDs
        # Sort tokens deterministically: special tokens first, then bytes, then alphabet/subwords
        sorted_tokens = sorted(
            final_vocab.keys(),
            key=lambda t: (
                0 if t.startswith("<|") and t.endswith("|>") else (1 if t.startswith("<0x") else 2),
                -len(t),
                t,
            ),
        )

        token_to_id = {tok: idx for idx, tok in enumerate(sorted_tokens)}
        id_to_token = {idx: tok for idx, tok in enumerate(sorted_tokens)}

        return UnigramModel(
            vocab=final_vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special_tokens=list(seed_builder.special_tokens),
            max_subword_len=self.max_ngram_length,
            byte_fallback=self.byte_fallback,
        )
