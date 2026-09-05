from __future__ import annotations

import json
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

from .bpe_model import BPEModel
from .byte_codec import ByteFallbackEngine, validate_dropout_prob as _validate_dropout_prob
from .indentation_compressor import IndentationCompressor
from .pre_tokenizer import Normalizer, RegexPreTokenizer
from .security_shield import SecurityShield
from .seed_builder import SeedVocabularyBuilder
from .streaming_decoder import StreamingDecoder
from .unigram_trainer import UnigramModel, UnigramTrainer

# Native Rust core, preferring the repo's own crate name. Kept at module level
# so inner functions never `import caliper_core` (the stale site-packages
# module has incompatible class identity).
try:
    import uniqtoken_core as _native_core
except ImportError:
    try:
        import caliper_core as _native_core  # type: ignore[no-redef]
    except ImportError:
        _native_core = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Token:
    """
    Final Token emitted by the Tokenizer with token string, integer ID, and raw text offsets.
    """

    text: str
    id: int
    raw_span: Tuple[int, int]


@dataclass(frozen=True)
class TokenizationReport:
    """
    Detailed runtime diagnostic metrics emitted during tokenization.
    """

    tokens: List[str]
    token_ids: List[int]
    token_spans: List[Tuple[int, int]]
    num_tokens: int
    num_bytes: int
    num_chars: int
    byte_fallback_tokens: int
    byte_fallback_rate: float
    compression_ratio_bytes_per_token: float
    avg_token_length: float


class CustomTokenizer:
    """
    Production-Grade Byte-Fallback Unigram Custom Tokenizer.
    """

    def __init__(
        self,
        normalizer: Normalizer,
        pre_tokenizer: RegexPreTokenizer,
        model: UnigramModel,
    ):
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer
        self.model = model
        self.security = SecurityShield(special_tokens=self.model.special_tokens)
        self._cross_word_set: Optional[frozenset[str]] = None
        self._cross_word_model_id: Optional[int] = id(self.model)
        self._specials_pipe_form: Optional[bool] = None
        self._specials_model_id: Optional[int] = None

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    @staticmethod
    def _span(entry: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        return entry if isinstance(entry, tuple) else (entry, entry + 1)

    @classmethod
    def _compose_alignment(
        cls,
        inner_alignment: Sequence[Union[int, Tuple[int, int]]],
        outer_alignment: Sequence[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        composed: List[Tuple[int, int]] = []
        for entry in inner_alignment:
            start, end = cls._span(entry)
            source_spans = outer_alignment[start:end]
            if not source_spans:
                raise ValueError("alignment contains an empty source span")
            composed.append(
                (
                    min(span[0] for span in source_spans),
                    max(span[1] for span in source_spans),
                )
            )
        return composed

    def _cross_word_tokens(self) -> frozenset[str]:
        """Vocab tokens containing the space char (SuperBPE spanning tokens).

        These can never be emitted by the per-chunk Unigram lattice (a chunk is
        ``"the"`` or ``"\u2581"``, never ``"the\u2581quick"``), so they are only
        reachable through the post-encode merge pass.

        The result is cached, but invalidated whenever the model object changes
        so a reassigned ``tokenizer.model`` (e.g. after a CEM/SuperBPE vocabulary
        swap without constructing a fresh tokenizer) cannot leave the cache stale.
        """
        if self._cross_word_set is None or self._cross_word_model_id != id(self.model):
            sc = self.normalizer.space_char
            # Only internal metaspace (SuperBPE merges like "the▁quick" or "▁the▁quick"); leading
            # metaspace tokens (e.g. "▁quick" from pre-tokenization) are normal chunks.
            self._cross_word_set = frozenset(t for t in self.model.vocab if sc in t[1:] and t.strip(sc))
            self._cross_word_model_id = id(self.model)
        return self._cross_word_set

    def _special_tokens_contain_pipe_marker(self) -> bool:
        """True when every special token contains the literal ``<|``.

        This is the exact condition under which text containing a special token
        necessarily contains ``<|`` in its NFKC-canonical form — which is what
        the native pipeline's security gate keys on. Cached per model object
        like ``_cross_word_tokens``.
        """
        if self._specials_pipe_form is None or self._specials_model_id != id(self.model):
            self._specials_pipe_form = all("<|" in tok for tok in self.model.special_tokens)
            self._specials_model_id = id(self.model)
        return self._specials_pipe_form

    def _native_pipeline_kwargs(self) -> Optional[Dict[str, Any]]:
        """Kwargs for the fused native pipeline, or None when any configured
        stage would diverge from the native implementation.

        The native path replaces the Python SecurityShield/normalizer/
        pre-tokenizer/Viterbi stages with one Rust call, so it is only taken
        when every stage is provably equivalent:
        - no indent compression, no SuperBPE cross-word merges
        - all special tokens contain ``<|`` (see _special_tokens_contain_pipe_marker)
        - normalizer config expressible by the native normalizer (no casefold)
        - pre-tokenizer config exactly matches the native regex
        - the native side additionally refuses any text whose NFKC form could
          contain control-token syntax, falling back to the full Python path
        """
        if _native_core is None or not hasattr(_native_core, "rust_encode_text_native"):
            return None
        if self._indent_compression_enabled or self._cross_word_tokens():
            return None
        if not self._special_tokens_contain_pipe_marker():
            return None
        normalizer = self.normalizer
        if normalizer.casefold or not self.pre_tokenizer._native_pretok_parity:
            return None
        return {
            "space_char": normalizer.space_char,
            "normalize_unicode": normalizer.normalize_unicode,
            "normalize_unicode_spaces": normalizer.normalize_unicode_spaces,
            "normalize_punctuation": normalizer.normalize_punctuation,
            "lowercase": normalizer.lowercase,
            "collapse_whitespaces": normalizer.collapse_whitespaces,
            "strip_whitespace": normalizer.strip_whitespace,
        }

    def _encode_tokens_native_batch(self, texts: Sequence[str]) -> Optional[List[List[str]]]:
        """Batch-encode via the fused native pipeline (one FFI, Rayon across
        texts). Returns None whenever the caller must use the Python pipeline."""
        kwargs = self._native_pipeline_kwargs()
        if kwargs is None or not hasattr(_native_core, "rust_encode_text_native_batch"):
            return None
        assert _native_core is not None
        rust_trie = self.model._get_rust_trie()
        if rust_trie is None:
            return None
        try:
            return _native_core.rust_encode_text_native_batch(
                list(texts), rust_trie, self.model.byte_fallback, **kwargs
            )
        except (ValueError, TypeError, AttributeError):
            return None

    def _apply_cross_word_merges(self, tokens: List[str], dropout_prob: float = 0.0) -> List[str]:
        """Greedily fuses adjacent tokens until no SuperBPE merge remains.

        With ``dropout_prob > 0`` each candidate merge is independently
        skipped with that probability (SuperBPE merge dropout), which keeps
        the constituent tokens intact so stochastic regularization reaches
        cross-word merges as well. A boundary dropped by dropout stays
        blocked for the rest of this call: later merge passes never retry
        it unless one of its constituent pieces changes (e.g. by absorbing
        a neighbour), so the drawn segmentation cannot depend on unrelated
        merges elsewhere in the sequence.
        """
        cross = self._cross_word_tokens()
        if not cross:
            return tokens
        current = tokens
        # Positions (in ``current``) whose right-hand merge boundary was
        # dropped by dropout. They persist across passes — remapped as
        # elements shift — and are only released when one of the two
        # constituent pieces changes (absorbed by a neighbouring merge).
        blocked: Set[int] = set()
        while True:
            merged: List[str] = []
            changed = False
            i = 0
            remap: Dict[int, int] = {}  # old pass position -> position in ``merged``
            while i < len(current):
                m = len(merged)
                remap[i] = m
                if i + 1 < len(current) and i not in blocked and current[i] + current[i + 1] in cross:
                    if dropout_prob > 0.0 and random.random() < dropout_prob:
                        merged.append(current[i])
                        blocked.add(i)
                        i += 1
                    else:
                        merged.append(current[i] + current[i + 1])
                        remap[i + 1] = m
                        # Elements i and i+1 fuse: boundary (i, i+1) disappears
                        # and the neighbouring boundaries (i-1, i) / (i+1, i+2)
                        # gain a changed constituent, so unblock all three.
                        blocked.difference_update((i - 1, i, i + 1))
                        i += 2
                        changed = True
                else:
                    merged.append(current[i])
                    i += 1
            if not changed:
                return merged
            current = merged
            blocked = {remap[j] for j in blocked}

    def _apply_cross_word_merges_with_spans(self, tokens: List[Token], dropout_prob: float = 0.0) -> List[Token]:
        """Span-preserving counterpart of :meth:`_apply_cross_word_merges`.

        Merged tokens carry the union of their constituents' raw spans, and
        boundaries dropped by ``dropout_prob > 0`` stay blocked for the rest
        of the call exactly as in :meth:`_apply_cross_word_merges`.
        """
        cross = self._cross_word_tokens()
        if not cross:
            return tokens
        current = tokens
        # Positions (in ``current``) whose right-hand merge boundary was
        # dropped by dropout; same persistence/remap rules as in
        # :meth:`_apply_cross_word_merges`.
        blocked: Set[int] = set()
        while True:
            merged: List[Token] = []
            changed = False
            i = 0
            remap: Dict[int, int] = {}  # old pass position -> position in ``merged``
            while i < len(current):
                m = len(merged)
                remap[i] = m
                if i + 1 < len(current) and i not in blocked and current[i].text + current[i + 1].text in cross:
                    if dropout_prob > 0.0 and random.random() < dropout_prob:
                        merged.append(current[i])
                        blocked.add(i)
                        i += 1
                        continue
                    a, b = current[i], current[i + 1]
                    text = a.text + b.text
                    merged.append(
                        Token(
                            text=text,
                            id=self.model.token_to_id[text],
                            raw_span=(a.raw_span[0], b.raw_span[1]),
                        )
                    )
                    remap[i + 1] = m
                    blocked.difference_update((i - 1, i, i + 1))
                    i += 2
                    changed = True
                else:
                    merged.append(current[i])
                    i += 1
            if not changed:
                return merged
            current = merged
            blocked = {remap[j] for j in blocked}

    def _prepare_text(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]],
        disallowed_special_action: str,
    ) -> str:
        sanitized = self.security.sanitize(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        if self._indent_compression_enabled:
            return IndentationCompressor.compress_indents(sanitized, vocab=self.model.vocab)
        return sanitized

    def _prepare_text_with_alignment(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]],
        disallowed_special_action: str,
    ) -> Tuple[str, List[Tuple[int, int]]]:
        sanitized, sanitized_alignment = self.security.sanitize_with_alignment(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        if not self._indent_compression_enabled:
            return sanitized, sanitized_alignment

        compressed, compressed_alignment = IndentationCompressor.compress_indents_with_alignment(
            sanitized, vocab=self.model.vocab
        )
        return compressed, self._compose_alignment(compressed_alignment, sanitized_alignment)

    @classmethod
    def train_from_corpus(
        cls,
        corpus: List[str],
        target_vocab_size: int = 8000,
        seed_multiplier: float = 3.0,
        max_ngram_length: int = 16,
        min_frequency: int = 2,
        byte_fallback: bool = True,
        split_digits: bool = False,
        hex_literals: bool = True,
        digit_chunk_size: Optional[int] = None,
        digit_chunking: Literal["block3", "single", "greedy"] = "block3",
        preset: Optional[str] = None,
        special_tokens: Optional[List[str]] = None,
        compress_indents: bool = False,
        ranking_strategy: str = "char_savings",
        adaptive_multiplier: bool = False,
        max_edges_per_node: Optional[int] = None,
        min_edge_log_prob: Optional[float] = None,
        convergence_tolerance: float = 1e-4,
        script_balance_temperature: Optional[float] = None,
        min_boundary_entropy: Optional[float] = None,
        length_exponent: float = 1.0,
        pruning_length_exponent: float = 0.0,
        verbose: bool = True,
    ) -> CustomTokenizer:
        normalizer = Normalizer()
        pre_tokenizer = RegexPreTokenizer(
            split_digits=split_digits,
            hex_literals=hex_literals,
            digit_chunk_size=digit_chunk_size,
            digit_chunking=digit_chunking,
            preset=preset,
        )

        combined_special = (
            list(SeedVocabularyBuilder.DEFAULT_SPECIAL_TOKENS) if special_tokens is None else list(special_tokens)
        )
        if compress_indents:
            for it in IndentationCompressor.INDENT_SPECIAL_TOKENS:
                if it not in combined_special:
                    combined_special.append(it)

        chunks: List[str] = []
        for doc in corpus:
            if compress_indents:
                doc = IndentationCompressor.compress_indents(doc)
            norm = normalizer.normalize(doc)
            chunks.extend(pre_tokenizer.pre_tokenize(norm))

        if not chunks:
            raise ValueError(
                "Empty corpus: no pre-tokenized chunks were produced. "
                "Provide non-empty text (and disable compress_indents if it "
                "reduces everything to whitespace)."
            )

        trainer = UnigramTrainer(
            target_vocab_size=target_vocab_size,
            seed_multiplier=seed_multiplier,
            max_ngram_length=max_ngram_length,
            min_frequency=min_frequency,
            byte_fallback=byte_fallback,
            special_tokens=combined_special if combined_special else None,
            ranking_strategy=ranking_strategy,
            adaptive_multiplier=adaptive_multiplier,
            max_edges_per_node=max_edges_per_node,
            min_edge_log_prob=min_edge_log_prob,
            convergence_tolerance=convergence_tolerance,
            script_balance_temperature=script_balance_temperature,
            min_boundary_entropy=min_boundary_entropy,
            length_exponent=length_exponent,
            pruning_length_exponent=pruning_length_exponent,
        )

        model = trainer.train(chunks, verbose=verbose)
        return cls(normalizer=normalizer, pre_tokenizer=pre_tokenizer, model=model)

    def encode(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
        dropout_prob: float = 0.0,
    ) -> List[str]:
        """
        Tokenizes input text into string tokens with byte fallback.

        Args:
            text: Input text to tokenize.
            allowed_special: Which special tokens are allowed in ``text``.
            disallowed_special_action: Action for disallowed special tokens.
            dropout_prob: Probability of independently skipping each
                candidate SuperBPE cross-word merge (merge dropout,
                Provilkov et al. 2020). ``0.0`` (default) keeps
                tokenization fully deterministic; values in ``(0.0, 1.0)``
                use the Python path only — the native Rust fast path is
                bypassed because it cannot reproduce Python's RNG. A
                dropped merge boundary is final for the call and is never
                retried by later merge passes.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        _validate_dropout_prob(dropout_prob)
        if not text:
            return []

        # Fused native fast path: sanitize (identity here), normalization,
        # pre-tokenization and Viterbi all happen in ONE FFI call. The Rust
        # side re-checks the security gate (NFKC-canonical "<|") and raises,
        # falling back to the full Python pipeline below. Skipped when merge
        # dropout is active: the native core cannot reproduce Python's RNG.
        native_kwargs = self._native_pipeline_kwargs()
        if (
            native_kwargs is not None
            and dropout_prob == 0.0
            and allowed_special == "none"
            and disallowed_special_action == "escape"
        ):
            assert _native_core is not None
            rust_trie = self.model._get_rust_trie()
            if rust_trie is None:
                native_kwargs = None
            else:
                try:
                    return _native_core.rust_encode_text_native(
                        text, rust_trie, self.model.byte_fallback, **native_kwargs
                    )
                except (ValueError, TypeError, AttributeError):
                    pass  # control-token syntax or unavailable native core → Python path

        sanitized_text = self._prepare_text(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )

        norm = self.normalizer.normalize(sanitized_text)
        chunks = self.pre_tokenizer.pre_tokenize(norm)

        # ponytail: batch per-text chunks 46→1 call; keep special-token bypass
        all_tokens: List[str] = []
        if not chunks:
            return []
        special = set(self.model.special_tokens)
        # fast path: no specials in this text (common)
        if not any(c in special for c in chunks):
            for lst in self.model.encode_batch(chunks):
                all_tokens.extend(lst)
        else:
            # mixed — batch only non-specials
            non_special = [c for c in chunks if c not in special]
            batch = self.model.encode_batch(non_special) if non_special else []
            it = iter(batch)
            for c in chunks:
                if c in special:
                    all_tokens.append(c)
                else:
                    all_tokens.extend(next(it))

        return self._apply_cross_word_merges(all_tokens, dropout_prob=dropout_prob)

    def sample(
        self,
        text: str,
        alpha: float = 0.5,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
        dropout_prob: float = 0.0,
    ) -> List[str]:
        """Sample tokenization using Subword Regularization (Kudo 2018).

        Args:
            text: Input text to tokenize.
            alpha: Sampling temperature for the lattice sampler.
            allowed_special: Which special tokens are allowed in ``text``.
            disallowed_special_action: Action for disallowed special tokens.
            dropout_prob: Merge-dropout probability applied to SuperBPE
                cross-word merges; behaves as in :meth:`encode`.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        _validate_dropout_prob(dropout_prob)
        if not text:
            return []
        if alpha <= 0:
            raise ValueError(f"alpha must be greater than zero, got {alpha}")

        sanitized_text = self._prepare_text(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        norm = self.normalizer.normalize(sanitized_text)
        chunks = self.pre_tokenizer.pre_tokenize(norm)

        all_tokens: List[str] = []
        for chunk in chunks:
            if chunk in self.model.special_tokens:
                all_tokens.append(chunk)
            else:
                all_tokens.extend(self.model.sample(chunk, alpha=alpha))

        return self._apply_cross_word_merges(all_tokens, dropout_prob=dropout_prob)

    def encode_to_ids(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
        dropout_prob: float = 0.0,
    ) -> List[int]:
        """Encodes text to token IDs; ``dropout_prob`` behaves as in :meth:`encode`."""
        tokens = self.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
            dropout_prob=dropout_prob,
        )
        unk_id = self.model.token_to_id.get(self.model.unk_token, 0)
        return [self.model.token_to_id.get(t, unk_id) for t in tokens]

    def sample_to_ids(
        self,
        text: str,
        alpha: float = 0.5,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
        dropout_prob: float = 0.0,
    ) -> List[int]:
        tokens = self.sample(
            text,
            alpha=alpha,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
            dropout_prob=dropout_prob,
        )
        unk_id = self.model.token_to_id.get(self.model.unk_token, 0)
        return [self.model.token_to_id.get(t, unk_id) for t in tokens]

    def encode_with_offsets(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
        dropout_prob: float = 0.0,
    ) -> List[Token]:
        """Encodes text to tokens with exact character spans.

        Merges dropped by ``dropout_prob > 0`` keep their constituents'
        spans intact, so the emitted spans always tile the input without
        gaps or overlaps and decode reconstruction is preserved.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        _validate_dropout_prob(dropout_prob)
        if not text:
            return []

        prepared_text, prepared_alignment = self._prepare_text_with_alignment(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        norm, normalization_alignment = self.normalizer.normalize_with_alignment(prepared_text)
        alignment = self._compose_alignment(normalization_alignment, prepared_alignment)
        pre_tokens = self.pre_tokenizer.pre_tokenize_with_offsets(norm, alignment)

        result: List[Token] = []
        unk_id = self.model.token_to_id.get(self.model.unk_token, 0)

        for pt in pre_tokens:
            chunk = pt.text
            if chunk in self.model.special_tokens:
                t_id = self.model.token_to_id.get(chunk, unk_id)
                result.append(Token(text=chunk, id=t_id, raw_span=pt.raw_span))
            else:
                for st, start, end in self.model.encode_with_spans(chunk):
                    t_id = self.model.token_to_id.get(st, unk_id)
                    norm_start = pt.norm_span[0] + start
                    norm_end = pt.norm_span[0] + end
                    source_spans = alignment[norm_start:norm_end]
                    raw_span = (
                        min(span[0] for span in source_spans),
                        max(span[1] for span in source_spans),
                    )
                    result.append(Token(text=st, id=t_id, raw_span=raw_span))

        return self._apply_cross_word_merges_with_spans(result, dropout_prob=dropout_prob)

    def encode_batch(
        self,
        texts: Sequence[str],
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
        num_workers: Optional[int] = None,
        dropout_prob: float = 0.0,
    ) -> List[List[str]]:
        """Encodes a sequence of texts, parallelizing across workers when batch is large.

        ``dropout_prob`` is honored on every row; non-zero values bypass the
        native Rust fused batch path and use the per-text Python path
        (the native core cannot reproduce Python's RNG). With
        ``dropout_prob > 0`` and ``num_workers > 1``, rows draw from the
        shared global ``random`` module under thread scheduling, so
        per-row segmentations are not reproducible across runs even with
        a fixed seed.
        """
        _validate_dropout_prob(dropout_prob)
        if not texts:
            return []
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"num_workers must be >= 1 (or None), got {num_workers}")
        if len(texts) <= 64 or num_workers == 1:
            return [
                self.encode(
                    t,
                    allowed_special=allowed_special,
                    disallowed_special_action=disallowed_special_action,
                    dropout_prob=dropout_prob,
                )
                for t in texts
            ]

        workers = num_workers or min(os.cpu_count() or 1, 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda t: self.encode(
                        t,
                        allowed_special=allowed_special,
                        disallowed_special_action=disallowed_special_action,
                        dropout_prob=dropout_prob,
                    ),
                    texts,
                )
            )

    def encode_to_ids_batch(
        self,
        texts: Sequence[str],
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
        num_workers: Optional[int] = None,
        dropout_prob: float = 0.0,
    ) -> List[List[int]]:
        """Encodes a sequence of texts to token IDs, parallelizing across workers when batch is large.

        Non-zero ``dropout_prob`` bypasses the native Rust fused batch path
        and encodes each text through the Python path. With
        ``dropout_prob > 0`` and ``num_workers > 1``, per-row results are
        not reproducible across runs because rows share the global RNG.
        """
        _validate_dropout_prob(dropout_prob)
        if not texts:
            return []
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"num_workers must be >= 1 (or None), got {num_workers}")

        # Fused native batch: one FFI + Rayon. Only when the whole batch can be
        # proven equivalent to the per-text Python path (gates below). Skipped
        # when dropout is active: the native core cannot reproduce Python's RNG.
        if (
            allowed_special == "none"
            and disallowed_special_action == "escape"
            and (num_workers is None or num_workers > 1)
            and dropout_prob == 0.0
        ):
            native_tokens = self._encode_tokens_native_batch(texts)
            if native_tokens is not None:
                token_to_id = self.model.token_to_id
                unk_id = token_to_id.get(self.model.unk_token, 0)
                return [[token_to_id.get(tok, unk_id) for tok in toks] for toks in native_tokens]

        if len(texts) <= 64 or num_workers == 1:
            return [
                self.encode_to_ids(
                    t,
                    allowed_special=allowed_special,
                    disallowed_special_action=disallowed_special_action,
                    dropout_prob=dropout_prob,
                )
                for t in texts
            ]

        workers = num_workers or min(os.cpu_count() or 1, 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda t: self.encode_to_ids(
                        t,
                        allowed_special=allowed_special,
                        disallowed_special_action=disallowed_special_action,
                        dropout_prob=dropout_prob,
                    ),
                    texts,
                )
            )

    def encode_with_offsets_batch(
        self,
        texts: Sequence[str],
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
        num_workers: Optional[int] = None,
        dropout_prob: float = 0.0,
    ) -> List[List[Token]]:
        """Encodes a sequence of texts with exact spans, parallelizing across workers when batch is large.

        Merges dropped by non-zero ``dropout_prob`` keep their constituents'
        spans intact, so spans tile each input without gaps or overlaps.
        With ``dropout_prob > 0`` and ``num_workers > 1``, per-row results
        are not reproducible across runs because rows share the global RNG.
        """
        _validate_dropout_prob(dropout_prob)
        if not texts:
            return []
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"num_workers must be >= 1 (or None), got {num_workers}")
        if len(texts) <= 64 or num_workers == 1:
            return [
                self.encode_with_offsets(
                    t,
                    allowed_special=allowed_special,
                    disallowed_special_action=disallowed_special_action,
                    dropout_prob=dropout_prob,
                )
                for t in texts
            ]

        workers = num_workers or min(os.cpu_count() or 1, 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda t: self.encode_with_offsets(
                        t,
                        allowed_special=allowed_special,
                        disallowed_special_action=disallowed_special_action,
                        dropout_prob=dropout_prob,
                    ),
                    texts,
                )
            )

    def encode_with_metrics(
        self,
        text: str,
        allowed_special: Union[str, Set[str], List[str]] = "none",
        disallowed_special_action: str = "escape",
    ) -> TokenizationReport:
        """
        Encodes text and computes runtime diagnostic metrics (byte-fallback rate, compression ratio).
        """
        tokens_with_offsets = self.encode_with_offsets(
            text,
            allowed_special=allowed_special,
            disallowed_special_action=disallowed_special_action,
        )
        tokens = [t.text for t in tokens_with_offsets]
        token_ids = [t.id for t in tokens_with_offsets]
        token_spans = [t.raw_span for t in tokens_with_offsets]
        raw_bytes = text.encode("utf-8")
        num_tokens = len(tokens)
        num_bytes = len(raw_bytes)
        num_chars = len(text)

        byte_fallback_tokens = sum(1 for t in tokens if ByteFallbackEngine.is_byte_token(t))
        byte_fallback_rate = (byte_fallback_tokens / num_tokens) if num_tokens > 0 else 0.0
        compression_ratio = (num_bytes / num_tokens) if num_tokens > 0 else 0.0
        avg_token_len = (sum(len(t) for t in tokens) / num_tokens) if num_tokens > 0 else 0.0

        return TokenizationReport(
            tokens=tokens,
            token_ids=token_ids,
            token_spans=token_spans,
            num_tokens=num_tokens,
            num_bytes=num_bytes,
            num_chars=num_chars,
            byte_fallback_tokens=byte_fallback_tokens,
            byte_fallback_rate=byte_fallback_rate,
            compression_ratio_bytes_per_token=compression_ratio,
            avg_token_length=avg_token_len,
        )

    @property
    def _indent_compression_enabled(self) -> bool:
        return any(tok in self.model.special_tokens for tok in IndentationCompressor.INDENT_SPECIAL_TOKENS)

    def decode(self, token_ids: List[int]) -> str:
        decoded = self.model.decode(token_ids, space_char=self.normalizer.space_char)
        if self._indent_compression_enabled:
            decoded = IndentationCompressor.decompress_indents(decoded)
        return self.normalizer.restore_escaped_metaspace(decoded)

    def decode_tokens(self, tokens: Sequence[str]) -> str:
        """Decodes a list of token strings directly back to the original text string."""
        unk_id = self.model.token_to_id.get(self.model.unk_token, 0)
        token_ids = [self.model.token_to_id.get(t, unk_id) for t in tokens]
        return self.decode(token_ids)

    def decode_batch(
        self,
        token_id_sequences: Sequence[Sequence[int]],
        num_workers: Optional[int] = None,
    ) -> List[str]:
        """Decodes a batch of token-ID sequences in parallel.

        This is the batch counterpart of :meth:`encode_batch`. It
        delegates to :meth:`decode` for each row; for large batches a
        thread pool provides modest speedup because the inner
        :meth:`BPEModel.decode` is Python-bound and the GIL is
        released during the byte-fallback decoding step.
        """
        if not token_id_sequences:
            return []
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"num_workers must be >= 1 (or None), got {num_workers}")
        if len(token_id_sequences) <= 64 or num_workers == 1:
            return [self.decode(list(seq)) for seq in token_id_sequences]
        workers = num_workers or min(os.cpu_count() or 1, 8)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda seq: self.decode(list(seq)),
                    token_id_sequences,
                )
            )

    def get_streaming_decoder(self, skip_special_tokens: bool = True) -> StreamingDecoder:
        indent_replacements = {}
        if self._indent_compression_enabled:
            indent_replacements = {token: " " * count for count, token in IndentationCompressor.INDENT_MAP}
            indent_replacements["<|tab|>"] = "\t"
        return StreamingDecoder(
            id_to_token=self.model.id_to_token,
            space_char=self.normalizer.space_char,
            skip_special_tokens=skip_special_tokens,
            special_tokens=self.model.special_tokens,
            special_replacements=indent_replacements,
            metaspace_escape=(
                self.normalizer._ESCAPE_PREFIX,
                self.normalizer._ESCAPED_METASPACE,
            ),
        )

    def save(self, directory: Union[str, Path], save_binary: bool = True) -> None:
        """Saves tokenizer configuration and vocabulary. Automatically generates .uniqtok binary format."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        config = {
            "vocab": self.model.vocab,
            "token_to_id": self.model.token_to_id,
            "special_tokens": self.model.special_tokens,
            "space_char": self.normalizer.space_char,
            "split_digits": self.pre_tokenizer.split_digits,
            "max_subword_len": self.model.max_subword_len,
            "byte_fallback": self.model.byte_fallback,
            "unk_token": self.model.unk_token,
            "normalizer": {
                "space_char": self.normalizer.space_char,
                "lowercase": self.normalizer.lowercase,
                "casefold": self.normalizer.casefold,
                "normalize_unicode": self.normalizer.normalize_unicode,
                "normalize_punctuation": self.normalizer.normalize_punctuation,
                "normalize_unicode_spaces": self.normalizer.normalize_unicode_spaces,
                "collapse_whitespaces": self.normalizer.collapse_whitespaces,
                "strip_whitespace": self.normalizer.strip_whitespace,
            },
            "pre_tokenizer": {
                "space_char": self.pre_tokenizer.space_char,
                "split_digits": self.pre_tokenizer.split_digits,
                "split_punctuation": self.pre_tokenizer.split_punctuation,
                "keep_special_tokens": self.pre_tokenizer.keep_special_tokens,
                "special_token_pattern": self.pre_tokenizer.special_token_pattern,
                "hex_literals": self.pre_tokenizer.hex_literals,
                "digit_chunk_size": self.pre_tokenizer.digit_chunk_size,
                "digit_chunking": self.pre_tokenizer.digit_chunking,
                "preset": self.pre_tokenizer.preset,
            },
        }

        with open(dir_path / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        if save_binary:
            from uniqtoken.binary_format import export_binary

            binary_path = dir_path / "tokenizer.uniqtok"
            try:
                export_binary(self, binary_path)
            except ValueError:
                # Sparse token IDs or inconsistent vocab cannot be packed in contiguous binary format;
                # preserve successful JSON save and remove any stale binary file.
                if binary_path.is_file():
                    binary_path.unlink()

    @classmethod
    def load(cls, directory: Union[str, Path], prefer_binary: bool = True) -> CustomTokenizer:
        """Loads a CustomTokenizer. Prefers zero-copy .uniqtok format for sub-millisecond cold starts."""
        dir_path = Path(directory)
        binary_file = dir_path / "tokenizer.uniqtok"

        if prefer_binary and binary_file.is_file():
            try:
                from uniqtoken.binary_format import load_binary

                return load_binary(binary_file, use_mmap=True)
            except (ValueError, OSError) as e:
                # Safe fallback to standard JSON on corrupted binary or I/O failure
                warnings.warn(
                    f"Failed to load binary model from {binary_file} ({e}); falling back to JSON format.",
                    UserWarning,
                    stacklevel=2,
                )

        with open(dir_path / "tokenizer.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        vocab = config["vocab"]
        token_to_id = config["token_to_id"]
        id_to_token = {int(v): k for k, v in token_to_id.items()}
        special_tokens = config["special_tokens"]

        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special_tokens=special_tokens,
            max_subword_len=config.get("max_subword_len", 16),
            byte_fallback=config.get("byte_fallback", True),
            unk_token=config.get("unk_token", "<|unk|>"),
        )

        normalizer_config = config.get("normalizer", {})
        pre_tokenizer_config = config.get("pre_tokenizer", {})

        normalizer = Normalizer(
            space_char=normalizer_config.get("space_char", config.get("space_char", "\u2581")),
            lowercase=normalizer_config.get("lowercase", False),
            casefold=normalizer_config.get("casefold", False),
            normalize_unicode=normalizer_config.get("normalize_unicode", True),
            normalize_punctuation=normalizer_config.get("normalize_punctuation", False),
            normalize_unicode_spaces=normalizer_config.get("normalize_unicode_spaces", True),
            collapse_whitespaces=normalizer_config.get("collapse_whitespaces", False),
            strip_whitespace=normalizer_config.get("strip_whitespace", False),
        )
        pre_tokenizer = RegexPreTokenizer(
            space_char=pre_tokenizer_config.get("space_char", config.get("space_char", "\u2581")),
            split_digits=pre_tokenizer_config.get("split_digits", config.get("split_digits", False)),
            split_punctuation=pre_tokenizer_config.get("split_punctuation", True),
            keep_special_tokens=pre_tokenizer_config.get("keep_special_tokens", True),
            special_token_pattern=pre_tokenizer_config.get("special_token_pattern", r"<\|[^\s|]+\|>"),
            hex_literals=pre_tokenizer_config.get("hex_literals", True),
            digit_chunk_size=pre_tokenizer_config.get("digit_chunk_size"),
            # Legacy configs predate digit_chunking: they trained with greedy
            # digits (digit_chunk_size=None) or an explicit chunk size.
            digit_chunking=pre_tokenizer_config.get(
                "digit_chunking",
                "greedy" if pre_tokenizer_config.get("digit_chunk_size") is None else "block3",
            ),
            preset=pre_tokenizer_config.get("preset"),
        )

        return cls(normalizer=normalizer, pre_tokenizer=pre_tokenizer, model=model)

    def export_to_huggingface(self, directory: Union[str, Path]) -> None:
        """
        Exports the tokenizer to canonical HuggingFace tokenizer.json and tokenizer_config.json schema.
        """
        from .hf_exporter import HuggingFaceExporter

        HuggingFaceExporter.save_hf_pretrained(self, directory)

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        commit_message: str = "Upload UniqToken model",
        private: bool = False,
        **kwargs: Any,
    ) -> str:
        """Pushes the HuggingFace-compatible tokenizer files to the Hugging Face Hub.

        Args:
            repo_id: Hub repository id of the form ``"owner/model"``.
            token: Optional Hub access token used for authentication.
            commit_message: Commit message recorded for the upload commit.
            private: Repository visibility applied when the repo is created. Has no
                effect on an already existing repo.
            **kwargs: Forwarded to ``HfApi.upload_folder``.

        Returns:
            The commit URL of the completed synchronous upload, or the PR URL
            string when ``multi_commits=True`` is passed.
        """
        from .hf_exporter import HuggingFaceExporter

        return HuggingFaceExporter.push_to_hub(
            self, repo_id, token=token, commit_message=commit_message, private=private, **kwargs
        )

    def export_to_gguf(self, output_path: Optional[Union[str, Path]] = None, model_name: str = "llama") -> bytes:
        """
        Exports the tokenizer to LLaMA.cpp GGUF v3 binary format.
        Optionally writes to output_path if provided, and returns the binary GGUF bytes.
        """
        from .hf_exporter import HuggingFaceExporter

        return HuggingFaceExporter.export_to_gguf(self, output_path=output_path, model_name=model_name)
