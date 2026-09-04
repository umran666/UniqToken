from __future__ import annotations

import math
import regex as _regex
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

# Extended grapheme clusters (UAX #29) with current-Unicode data, mirroring
# the Rust engine's unicode-segmentation crate.
_X_RE = _regex.compile(r"\X")
_MARK_RE = _regex.compile(r"\p{M}")

# Module-level Rust core alias, preferring the repo's own crate name. Kept in
# one place so inner functions never `import caliper_core` (a stale
# site-packages module whose RustPrefixTrie class is a *different* type).
try:
    import uniqtoken_core as caliper_core
except ImportError:
    try:
        import caliper_core  # type: ignore[no-redef]
    except ImportError:
        caliper_core = None  # type: ignore[assignment]


def _grapheme_clusters(text: str) -> list[str]:
    """Split `text` into extended grapheme clusters (UAX #29)."""
    if _X_RE is not None:
        return [m.group(0) for m in _X_RE.finditer(text)]
    # Fallback without `regex`: every codepoint is its own cluster.
    return [ch for ch in text]


@dataclass(frozen=True)
class SeedToken:
    """
    Represents a token in the seed vocabulary.

    is_required: True for special tokens, byte fallbacks, and the base alphabet.
                 These tokens are IMMUNE to pruning during Unigram EM iterations.
    """

    token: str
    frequency: int
    is_required: bool
    source: str  # "special" | "byte" | "alphabet" | "ngram"
    length: int


class SeedVocabularyBuilder:
    """
    Deterministic Seed Vocabulary Builder with Irreducible Floor Validation,
    Pointwise Mutual Information (PMI) filtering, and Adaptive Pool Sizing.
    """

    DEFAULT_SPECIAL_TOKENS = [
        "<|pad|>",
        "<|unk|>",
        "<|bos|>",
        "<|eos|>",
        "<|endoftext|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
    ]

    def __init__(
        self,
        target_vocab_size: int = 8000,
        seed_multiplier: float = 3.0,
        max_ngram_length: int = 16,
        min_frequency: int = 2,
        byte_fallback: bool = True,
        special_tokens: List[str] | None = None,
        ranking_strategy: str = "char_savings",
        adaptive_multiplier: bool = False,
        script_balance_temperature: Optional[float] = None,
        min_boundary_entropy: Optional[float] = None,
        length_exponent: float = 1.0,
    ):
        if target_vocab_size <= 0:
            raise ValueError("target_vocab_size must be greater than zero")
        if seed_multiplier <= 0:
            raise ValueError("seed_multiplier must be greater than zero")
        if max_ngram_length < 1:
            raise ValueError("max_ngram_length must be at least one")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least one")
        if ranking_strategy not in {"char_savings", "byte_savings", "frequency", "pmi"}:
            raise ValueError("ranking_strategy must be 'char_savings', 'byte_savings', 'frequency', or 'pmi'")
        if script_balance_temperature is not None and script_balance_temperature <= 0:
            raise ValueError("script_balance_temperature must be greater than zero")
        if min_boundary_entropy is not None and min_boundary_entropy < 0:
            raise ValueError("min_boundary_entropy cannot be negative")
        self.target_vocab_size = target_vocab_size
        self.seed_multiplier = seed_multiplier
        self.adaptive_multiplier = adaptive_multiplier
        self.script_balance_temperature = script_balance_temperature
        self.min_boundary_entropy = min_boundary_entropy
        self.length_exponent = length_exponent
        self.seed_vocab_size = int(target_vocab_size * seed_multiplier)
        self.max_ngram_length = max_ngram_length
        self.min_frequency = min_frequency
        self.byte_fallback = byte_fallback
        self.special_tokens = special_tokens if special_tokens is not None else list(self.DEFAULT_SPECIAL_TOKENS)
        self.ranking_strategy = ranking_strategy

    def collect_special_tokens(self) -> List[SeedToken]:
        tokens: List[SeedToken] = []
        for token in self.special_tokens:
            tokens.append(
                SeedToken(
                    token=token,
                    frequency=1,
                    is_required=True,
                    source="special",
                    length=len(token),
                )
            )
        return tokens

    def collect_byte_tokens(self) -> List[SeedToken]:
        if not self.byte_fallback:
            return []

        tokens: List[SeedToken] = []
        for b in range(256):
            byte_repr = f"<0x{b:02X}>"
            tokens.append(
                SeedToken(
                    token=byte_repr,
                    frequency=1,
                    is_required=True,
                    source="byte",
                    length=len(byte_repr),
                )
            )
        return tokens

    @staticmethod
    def _is_combining_mark(char: str) -> bool:
        """Issue #41: True for combining marks (Mn/Mc/Me), current-Unicode aware."""
        return _MARK_RE.match(char) is not None

    def collect_base_alphabet(self, chunk_counts: Counter[str]) -> List[SeedToken]:
        char_counts: Counter[str] = Counter()
        for chunk, count in chunk_counts.items():
            for char in chunk:
                # Issue #41: never emit a standalone combining mark without its base.
                if self._is_combining_mark(char):
                    continue
                char_counts[char] += count

        # Deterministic sorting: frequency descending, then unicode codepoint ascending
        sorted_chars = sorted(char_counts.items(), key=lambda x: (-x[1], x[0]))

        tokens: List[SeedToken] = []
        for char, count in sorted_chars:
            tokens.append(
                SeedToken(
                    token=char,
                    frequency=max(count, 1),
                    is_required=True,
                    source="alphabet",
                    length=1,
                )
            )
        return tokens

    _INDIC_SCRIPTS = frozenset({"devanagari", "telugu", "tamil", "bengali", "indic_other"})

    @staticmethod
    def _script_family(script: str) -> str:
        """Collapses granular script labels into audit families (e.g. devanagari → indic)."""
        if script in SeedVocabularyBuilder._INDIC_SCRIPTS:
            return "indic"
        return script

    @staticmethod
    def _detect_script(token: str) -> str:
        """Categorizes a token into a script family based on Unicode codepoints."""
        for ch in token:
            cp = ord(ch)
            # Skip metaspace, space markers, and private use prefixes
            if cp in {0x2581, 0x0020, 0x0009, 0x000A, 0x000D} or (0xE000 <= cp <= 0xF8FF):
                continue
            if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
                return "latin"
            elif 0x0900 <= cp <= 0x097F:
                return "devanagari"
            elif 0x0C00 <= cp <= 0x0C7F:
                return "telugu"
            elif 0x0B80 <= cp <= 0x0BFF:
                return "tamil"
            elif 0x0980 <= cp <= 0x09FF:
                return "bengali"
            elif 0x0900 <= cp <= 0x0D7F:
                return "indic_other"
            elif (
                (0x4E00 <= cp <= 0x9FFF)
                or (0x3400 <= cp <= 0x4DBF)
                or (0x3040 <= cp <= 0x30FF)
                or (0xAC00 <= cp <= 0xD7AF)
            ):
                return "cjk"
            elif (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F):
                return "arabic"
            elif (0x0400 <= cp <= 0x04FF) or (0x0500 <= cp <= 0x052F):
                return "cyrillic"
            elif 0x0E00 <= cp <= 0x0E7F:
                return "thai"
            elif ch.isdigit() or token.startswith("0x") or token.startswith("SYS_"):
                return "numeric"
        return "symbol"

    @staticmethod
    def _get_max_ngram_for_chunk(chunk: str, default_max: int) -> int:
        """Determines script-appropriate maximum n-gram mining length."""
        script = SeedVocabularyBuilder._detect_script(chunk)
        if script == "cjk":
            return min(default_max, 4)
        return default_max

    def mine_ngrams(self, chunk_counts: Counter[str]) -> Counter[str]:
        if self.min_boundary_entropy is not None:
            counts, _ = self.mine_ngrams_with_entropy(chunk_counts)
            return counts
        # ponytail: Rust &str slice + AHashMap if available; Python fallback exact.
        # Reuse the module-level uniqtoken_core alias (imported as caliper_core);
        # importing the stale site-packages `caliper_core` here would bypass the
        # repo's own Rust core and break class-identity for shared types.
        core = caliper_core if caliper_core is not None else None
        if core is not None:
            try:
                if hasattr(core, "rust_mine_ngrams"):
                    rust_res = core.rust_mine_ngrams(
                        dict(chunk_counts),
                        self.max_ngram_length,
                        set(self.special_tokens) if self.special_tokens else None,
                    )
                    return Counter(rust_res)
            except (ImportError, AttributeError, ValueError, TypeError):
                pass
        ngram_counts: Counter[str] = Counter()
        default_max = self.max_ngram_length
        for chunk, chunk_freq in chunk_counts.items():
            if chunk in self.special_tokens or (chunk.startswith("<|") and chunk.endswith("|>")):
                continue
            clusters = _grapheme_clusters(chunk)
            cluster_len = len(clusters)
            max_len = self._get_max_ngram_for_chunk(chunk, default_max)
            for start in range(cluster_len):
                # Issue #41: never start an n-gram with an orphan combining mark.
                if self._is_combining_mark(clusters[start]):
                    continue
                end_limit = min(cluster_len + 1, start + max_len + 1)
                for end in range(start + 1, end_limit):
                    ngram_counts["".join(clusters[start:end])] += chunk_freq
        return ngram_counts

    def mine_ngrams_with_entropy(self, chunk_counts: Counter[str]) -> Tuple[Counter[str], Dict[str, float]]:
        ngram_counts: Counter[str] = Counter()
        left_ctx: Dict[str, Counter[str]] = {}
        right_ctx: Dict[str, Counter[str]] = {}
        default_max = self.max_ngram_length

        for chunk, chunk_freq in chunk_counts.items():
            if chunk in self.special_tokens or (chunk.startswith("<|") and chunk.endswith("|>")):
                continue

            clusters = _grapheme_clusters(chunk)
            cluster_len = len(clusters)
            # Precompute codepoint offsets for each cluster boundary,
            # so l_char / r_char can be looked up from the original string.
            offsets = [0]
            for c in clusters:
                offsets.append(offsets[-1] + len(c))
            chunk_len = len(chunk)
            max_len = self._get_max_ngram_for_chunk(chunk, default_max)
            for start in range(cluster_len):
                # Issue #41: never start an n-gram with an orphan combining mark.
                if self._is_combining_mark(clusters[start]):
                    continue
                end_limit = min(cluster_len + 1, start + max_len + 1)
                l_char = chunk[offsets[start] - 1] if offsets[start] > 0 else "^"
                for end in range(start + 1, end_limit):
                    sub = "".join(clusters[start:end])
                    r_char = chunk[offsets[end]] if offsets[end] < chunk_len else "$"
                    ngram_counts[sub] += chunk_freq
                    if self.min_boundary_entropy is not None:
                        if sub not in left_ctx:
                            left_ctx[sub] = Counter()
                            right_ctx[sub] = Counter()
                        left_ctx[sub][l_char] += chunk_freq
                        right_ctx[sub][r_char] += chunk_freq

        entropies: Dict[str, float] = {}
        if self.min_boundary_entropy is not None:
            for sub, counts in ngram_counts.items():
                l_cnt = left_ctx.get(sub, Counter())
                r_cnt = right_ctx.get(sub, Counter())
                l_tot = sum(l_cnt.values())
                r_tot = sum(r_cnt.values())
                h_l = -sum((c / l_tot) * math.log2(c / l_tot) for c in l_cnt.values() if c > 0) if l_tot > 0 else 0.0
                h_r = -sum((c / r_tot) * math.log2(c / r_tot) for c in r_cnt.values() if c > 0) if r_tot > 0 else 0.0
                entropies[sub] = min(h_l, h_r)

        return ngram_counts, entropies

    def filter_candidates(
        self,
        ngram_counts: Counter[str],
        protected_tokens: Set[str],
        entropies: Optional[Dict[str, float]] = None,
    ) -> Dict[str, int]:
        filtered: Dict[str, int] = {}
        for token, count in ngram_counts.items():
            if token in protected_tokens:
                continue
            if count < self.min_frequency:
                continue
            # Issue #41: belt-and-braces — drop orphan combining-mark candidates.
            if token and self._is_combining_mark(token[0]):
                continue
            if (
                self.min_boundary_entropy is not None
                and entropies is not None
                and len(token) > 1
                and entropies.get(token, 0.0) < self.min_boundary_entropy
            ):
                continue
            filtered[token] = count
        return filtered

    def rank_candidates(
        self,
        candidate_counts: Dict[str, int],
        unigram_counts: Optional[Dict[str, int]] = None,
        total_unigrams: Optional[int] = None,
    ) -> List[str]:
        """
        Deterministic Candidate Ranking with optional script balancing and length regularization:
        - "char_savings": (len(t) - 1) * freq
        - "byte_savings": freq * ln(1 + byte_len) (marginal subword compression with memorization penalty)
        - "frequency": raw freq
        - "pmi": Pointwise Mutual Information cohesion scaled by savings
        """
        # ponytail: cache script label once; _detect_script scans token O(|t|)
        script_of: Dict[str, str] = {}
        if candidate_counts:
            script_of = {t: self._detect_script(t) for t in candidate_counts}

        script_weights: Dict[str, float] = {}
        if self.script_balance_temperature is not None and candidate_counts:
            T = self.script_balance_temperature
            script_totals: Counter[str] = Counter()
            for t, cnt in candidate_counts.items():
                script_totals[script_of[t]] += cnt

            temp_totals = {s: (tot**T) for s, tot in script_totals.items()}
            sum_temp = sum(temp_totals.values())
            for s, tot in script_totals.items():
                prob_target = temp_totals[s] / max(sum_temp, 1e-9)
                script_weights[s] = prob_target / max(tot, 1e-9)

        if self.ranking_strategy == "pmi" and unigram_counts and total_unigrams and total_unigrams > 0:
            total_val = float(total_unigrams)
            log_tot = math.log(total_val)
            char_log_p = {c: math.log(max(cnt, 1)) - log_tot for c, cnt in unigram_counts.items()}

            def pmi_score(t: str) -> float:
                freq = candidate_counts[t]
                log_p_t = math.log(max(freq, 1)) - log_tot
                sum_char_log_p = sum(char_log_p.get(c, -10.0) for c in t)
                pmi = log_p_t - sum_char_log_p
                base = (len(t) - 1) * freq * max(0.1, pmi)
                if self.script_balance_temperature is not None:
                    return base * script_weights.get(script_of[t], 1.0)
                return base

            return sorted(
                candidate_counts.keys(),
                key=lambda t: (
                    -pmi_score(t),
                    -candidate_counts[t],
                    -len(t),
                    t,
                ),
            )
        elif self.ranking_strategy in {"char_savings", "byte_savings"}:
            is_byte = self.ranking_strategy == "byte_savings"
            # ponytail: cache byte_len to avoid encode per comparator call
            byte_len_of: Dict[str, int] = {}
            if is_byte:
                byte_len_of = {t: len(t.encode("utf-8")) for t in candidate_counts}

            def savings_score(t: str) -> float:
                if is_byte:
                    byte_len = byte_len_of[t]
                    effective_len = max(byte_len - 1, 1)
                    if self.length_exponent == 1.0:
                        base = float(candidate_counts[t]) * float(effective_len)
                    else:
                        base = float(candidate_counts[t]) * (float(effective_len) ** self.length_exponent)
                else:
                    char_len = max(len(t) - 1, 1)
                    if self.length_exponent == 1.0:
                        base = float(char_len * candidate_counts[t])
                    else:
                        base = float(candidate_counts[t]) * (float(char_len) ** self.length_exponent)

                if self.script_balance_temperature is not None:
                    return base * script_weights.get(script_of[t], 1.0)
                return base

            return sorted(
                candidate_counts.keys(),
                key=lambda t: (
                    -savings_score(t),
                    -candidate_counts[t],
                    -len(t),
                    t,
                ),
            )
        else:

            def freq_score(t: str) -> float:
                base = float(candidate_counts[t])
                if self.script_balance_temperature is not None:
                    return base * script_weights.get(script_of[t], 1.0)
                return base

            return sorted(
                candidate_counts.keys(),
                key=lambda t: (-freq_score(t), -candidate_counts[t], -len(t), t),
            )

    def build_seed_vocab(
        self, pre_tokenized_chunks: Iterable[str], enforce_target_floor: bool = True
    ) -> List[SeedToken]:
        """
        Assembles the complete Seed Vocabulary pool with script-stratified quota allocation.
        """
        chunk_counts: Counter[str] = Counter(pre_tokenized_chunks)
        seen_tokens: Set[str] = set()
        seed_vocab: List[SeedToken] = []

        # 1. Required Tokens
        for entry in self.collect_special_tokens():
            if entry.token not in seen_tokens:
                seed_vocab.append(entry)
                seen_tokens.add(entry.token)

        for entry in self.collect_byte_tokens():
            if entry.token not in seen_tokens:
                seed_vocab.append(entry)
                seen_tokens.add(entry.token)

        alphabet_tokens = self.collect_base_alphabet(chunk_counts)
        for entry in alphabet_tokens:
            if entry.token not in seen_tokens:
                seed_vocab.append(entry)
                seen_tokens.add(entry.token)

        num_required = len(seed_vocab)

        # 2. Floor Validation
        if enforce_target_floor and self.target_vocab_size < num_required:
            raise ValueError(
                f"target_vocab_size ({self.target_vocab_size}) is smaller than the required token floor ({num_required}). "
                f"Set target_vocab_size >= {num_required} or disable byte_fallback / reduce special tokens."
            )

        # Compute unigram statistics for PMI ranking if needed
        unigram_counts = {t.token: t.frequency for t in alphabet_tokens}
        total_unigrams = sum(unigram_counts.values())

        # 3. Mine, Filter, and Rank Candidates
        if self.min_boundary_entropy is not None:
            raw_ngrams, entropies = self.mine_ngrams_with_entropy(chunk_counts)
        else:
            raw_ngrams = self.mine_ngrams(chunk_counts)
            entropies = None
        filtered_candidates = self.filter_candidates(raw_ngrams, seen_tokens, entropies=entropies)
        ranked_candidates = self.rank_candidates(
            filtered_candidates,
            unigram_counts=unigram_counts,
            total_unigrams=total_unigrams,
        )

        # 4. Fill Seed Capacity with Script-Stratified Quotas
        if self.adaptive_multiplier:
            num_chunks = len(chunk_counts)
            scale = min(4.5, max(1.5, 1.0 + math.log10(max(num_chunks, 10))))
            effective_seed_size = max(num_required, int(self.target_vocab_size * scale))
        else:
            effective_seed_size = self.seed_vocab_size

        candidate_budget = max(0, effective_seed_size - len(seed_vocab))

        # Bucket candidates by script family
        script_buckets: Dict[str, Deque[str]] = {}
        for token in ranked_candidates:
            if token not in seen_tokens:
                s = self._detect_script(token)
                if s not in script_buckets:
                    script_buckets[s] = deque()
                script_buckets[s].append(token)

        # Interleave candidates across active scripts (round-robin stratified quotas)
        selected_candidates: List[str] = []
        if script_buckets:
            active_scripts = list(script_buckets.keys())
            # ponytail: deque popleft is O(1) vs list pop(0) O(n); upgrade to heap quotas if script skew matters
            remaining = sum(len(v) for v in script_buckets.values())
            idx = 0
            while len(selected_candidates) < candidate_budget and remaining > 0:
                s = active_scripts[idx % len(active_scripts)]
                bucket = script_buckets[s]
                if bucket:
                    selected_candidates.append(bucket.popleft())
                    remaining -= 1
                idx += 1
                # early exit when all buckets drained is covered by remaining==0

        for token in selected_candidates:
            seed_vocab.append(
                SeedToken(
                    token=token,
                    frequency=filtered_candidates[token],
                    is_required=False,
                    source="ngram",
                    length=len(token),
                )
            )
            seen_tokens.add(token)
            if len(seed_vocab) >= effective_seed_size:
                break

        return seed_vocab

    def stats(self, vocabulary: List[SeedToken]) -> dict:
        required = [t for t in vocabulary if t.is_required]
        candidates = [t for t in vocabulary if not t.is_required]
        return {
            "total": len(vocabulary),
            "required": len(required),
            "candidates": len(candidates),
            "target_vocab_size": self.target_vocab_size,
            "seed_vocab_size": self.seed_vocab_size,
            "max_ngram_length": self.max_ngram_length,
            "min_frequency": self.min_frequency,
            "sources": Counter(t.source for t in vocabulary),
        }
