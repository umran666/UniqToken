"""Descriptive tokenizer metrics over exact normalized source-byte spans."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field
from types import MappingProxyType
import unicodedata
from unittest.mock import patch

from uniqtoken.byte_codec import ByteFallbackEngine


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ratio(numerator, denominator, scale=1.0):
    return scale * numerator / denominator if denominator else None


def source_parts(tokenizer, ids):
    marker = getattr(getattr(tokenizer.model, "normalizer", None), "space_char", "\u2581")
    parts, fallback = [], []
    for token_id in ids:
        piece = tokenizer.piece_for_id(token_id)
        is_byte = ByteFallbackEngine.is_byte_token(piece)
        fallback.append(is_byte)
        parts.append(
            bytes([ByteFallbackEngine.token_to_byte(piece)]) if is_byte else piece.replace(marker, " ").encode("utf-8")
        )
    return parts, fallback


def encode_document(tokenizer, text):
    """Observe the production merge pass without changing its decisions or IDs."""
    calls = []
    if tokenizer.name == "uniq_superbpe":
        original = tokenizer.model._apply_cross_word_merges

        def observe(pieces, dropout_prob=0.0):
            require(dropout_prob == 0.0, "diagnostics require zero dropout")
            result = original(pieces, dropout_prob=dropout_prob)
            calls.append((len(pieces), len(result)))
            return result

        with patch.object(tokenizer.model, "_apply_cross_word_merges", observe):
            ids = tokenizer.encode(text)
        require(len(calls) == 1, "SuperBPE diagnostic did not observe exactly one production merge pass")
        initial, final = calls[0]
        require(final == len(ids), "merge observation/output mismatch")
    else:
        ids = tokenizer.encode(text)
        initial = len(ids)
    require(ids and all(type(i) is int and 4 <= i < len(tokenizer.vocab) for i in ids), "invalid diagnostic token IDs")
    parts, fallback = source_parts(tokenizer, ids)
    require(all(parts), "empty token source contribution")
    require(b"".join(parts) == text.encode("utf-8"), "token pieces do not reconstruct normalized source bytes")
    require(initial >= len(ids), "merge observation increased token count")
    return ids, parts, fallback, initial


def character_runs(text, predicate):
    """Maximal runs in source UTF-8 coordinates, retaining Unicode character counts."""
    runs, start, count, offset = [], None, 0, 0
    for char in text:
        if predicate(char):
            if start is None:
                start = offset
            count += 1
        elif start is not None:
            runs.append((start, offset, count))
            start, count = None, 0
        offset += len(char.encode("utf-8"))
    if start is not None:
        runs.append((start, offset, count))
    return runs


def fragmentation(runs, starts, ends):
    intersections, split_runs, excess = 0, 0, 0
    isolated: set[int] = set()
    mixed: set[int] = set()
    for start, end, _ in runs:
        left, right = bisect_right(ends, start), bisect_left(starts, end)
        count = right - left
        require(count > 0, "source run has no token coverage")
        intersections += count
        split_runs += count > 1
        excess += count - 1
        for index in range(left, right):
            (isolated if start <= starts[index] and ends[index] <= end else mixed).add(index)
    return {
        "runs": len(runs),
        "unicode_characters": sum(count for _, _, count in runs),
        "token_intersections": intersections,
        "split_runs": split_runs,
        "excess_fragments": excess,
        "isolated_tokens": len(isolated),
        "mixed_tokens": len(mixed),
    }


@dataclass
class Counts:
    documents: int = 0
    source_utf8_bytes: int = 0
    normalized_utf8_bytes: int = 0
    unicode_characters: int = 0
    initial_tokens: int = 0
    initial_adjacent_boundaries: int = 0
    applied_cross_word_merges: int = 0
    cross_word_tokens: int = 0
    byte_fallback_tokens: int = 0
    frequencies: Counter = field(default_factory=Counter)
    lengths: Counter = field(default_factory=Counter)
    boundaries: dict = field(default_factory=lambda: {kind: Counter() for kind in ("whitespace", "punctuation")})

    def observe(self, tokenizer, row, text):
        ids, parts, fallback, initial = encode_document(tokenizer, text)
        starts, ends, offset = [], [], 0
        for part in parts:
            starts.append(offset)
            offset += len(part)
            ends.append(offset)
        self.documents += 1
        self.source_utf8_bytes += row["raw_utf8_bytes"]
        self.normalized_utf8_bytes += offset
        self.unicode_characters += len(text)
        self.initial_tokens += initial
        self.initial_adjacent_boundaries += max(0, initial - 1)
        self.applied_cross_word_merges += initial - len(ids)
        self.byte_fallback_tokens += sum(fallback)
        self.frequencies.update(ids)
        self.lengths.update(map(len, parts))
        for kind, predicate in (
            ("whitespace", str.isspace),
            ("punctuation", lambda char: unicodedata.category(char).startswith("P")),
        ):
            self.boundaries[kind].update(fragmentation(character_runs(text, predicate), starts, ends))
        fields_per_token: Counter[int] = Counter()
        for start, end, _ in character_runs(text, lambda char: not char.isspace()):
            fields_per_token.update(range(bisect_right(ends, start), bisect_left(starts, end)))
        self.cross_word_tokens += sum(count > 1 for count in fields_per_token.values())

    def combine(self, other):
        for name in (
            "documents",
            "source_utf8_bytes",
            "normalized_utf8_bytes",
            "unicode_characters",
            "initial_tokens",
            "initial_adjacent_boundaries",
            "applied_cross_word_merges",
            "cross_word_tokens",
            "byte_fallback_tokens",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.frequencies.update(other.frequencies)
        self.lengths.update(other.lengths)
        for kind in self.boundaries:
            self.boundaries[kind].update(other.boundaries[kind])
        return self

    def metrics(self, vocabulary, training_frequencies, rare_threshold):
        tokens = sum(self.frequencies.values())
        usable = {token_id for token_id in vocabulary.values() if token_id >= 4}
        require(set(self.frequencies) <= usable, "observed ID missing from usable vocabulary")
        rare = sum(
            count for token_id, count in self.frequencies.items() if training_frequencies[token_id] <= rare_threshold
        )
        unseen = sum(count for token_id, count in self.frequencies.items() if training_frequencies[token_id] == 0)

        def quantile(probability):
            import math

            target, cumulative = math.ceil(probability * tokens), 0
            for length, count in sorted(self.lengths.items()):
                cumulative += count
                if cumulative >= target:
                    return length
            return None

        result = {
            "documents": self.documents,
            "source_utf8_bytes": self.source_utf8_bytes,
            "normalized_utf8_bytes": self.normalized_utf8_bytes,
            "unicode_characters": self.unicode_characters,
            "tokens": tokens,
            "bytes_per_token": ratio(self.normalized_utf8_bytes, tokens),
            "tokens_per_unicode_character": ratio(tokens, self.unicode_characters),
            "tokens_per_normalized_utf8_byte": ratio(tokens, self.normalized_utf8_bytes),
            "byte_fallback_tokens": self.byte_fallback_tokens,
            "byte_fallback_percent": ratio(self.byte_fallback_tokens, tokens, 100.0),
            "token_length_bytes_mean": ratio(self.normalized_utf8_bytes, tokens),
            "token_length_bytes_p50": quantile(0.5),
            "token_length_bytes_p95": quantile(0.95),
            "token_length_bytes_max": max(self.lengths, default=None),
            "token_length_bytes_histogram": {str(length): count for length, count in sorted(self.lengths.items())},
            "usable_vocabulary_size": len(usable),
            "observed_token_types": len(self.frequencies),
            "vocabulary_utilization_percent": ratio(len(self.frequencies), len(usable), 100.0),
            "rare_tokens": rare,
            "rare_token_percent": ratio(rare, tokens, 100.0),
            "unseen_in_diagnostic_training_tokens": unseen,
            "unseen_in_diagnostic_training_percent": ratio(unseen, tokens, 100.0),
            "initial_tokens": self.initial_tokens,
            "initial_adjacent_boundaries": self.initial_adjacent_boundaries,
            "applied_cross_word_merges": self.applied_cross_word_merges,
            "cross_word_merge_rate_percent": ratio(
                self.applied_cross_word_merges, self.initial_adjacent_boundaries, 100.0
            ),
            "cross_word_tokens": self.cross_word_tokens,
            "cross_word_token_percent": ratio(self.cross_word_tokens, tokens, 100.0),
        }
        for kind, counts in self.boundaries.items():
            result.update({f"{kind}_{name}": count for name, count in sorted(counts.items())})
            result[f"{kind}_tokens_per_run"] = ratio(counts["token_intersections"], counts["runs"])
            result[f"{kind}_split_run_percent"] = ratio(counts["split_runs"], counts["runs"], 100.0)
        require(sum(self.lengths.values()) == tokens, "token-length histogram accounting mismatch")
        require(
            sum(length * count for length, count in self.lengths.items()) == self.normalized_utf8_bytes,
            "byte accounting mismatch",
        )
        require(self.initial_tokens - tokens == self.applied_cross_word_merges, "merge accounting mismatch")
        return result


@contextmanager
def readonly_tokenizer(tokenizer):
    """Memoize invariant validation only while BPE lookup tables are immutable."""
    if tokenizer.name != "boundary_bpe":
        yield tokenizer
        return
    model = tokenizer.model
    model._validate_byte_fallback()
    snapshot = {
        "vocab": frozenset(model.vocab),
        "token_to_id": MappingProxyType(dict(model.token_to_id)),
        "id_to_token": MappingProxyType(dict(model.id_to_token)),
        "merges": MappingProxyType(dict(model.merges)),
    }
    with ExitStack() as stack:
        for name, value in snapshot.items():
            stack.enter_context(patch.object(model, name, value))
        stack.enter_context(patch.object(tokenizer, "vocab", snapshot["token_to_id"]))
        stack.enter_context(patch.object(model, "_validate_byte_fallback", lambda: None))
        yield tokenizer


def analyze_condition(tokenizer, assignments, rare_threshold=5):
    with readonly_tokenizer(tokenizer):
        return _analyze_condition(tokenizer, assignments, rare_threshold)


def _analyze_condition(tokenizer, assignments, rare_threshold):
    """Pool raw counts before computing ratios, vocabulary unions, and quantiles."""
    split_counts = {}
    for split in ("train", "validation"):
        strata: dict[tuple[str, str], Counts] = {}
        for row, text in assignments[split]:
            strata.setdefault((row["domain"], row["language"]), Counts()).observe(tokenizer, row, text)
        require(strata, f"no diagnostic documents for {split}")
        split_counts[split] = strata
    training_frequencies: Counter[int] = Counter()
    for counts in split_counts["train"].values():
        training_frequencies.update(counts.frequencies)
    records = []
    for split, strata in split_counts.items():
        scopes: list[tuple[str, str | None, str | None, Counts]] = [
            ("stratum", domain, language, counts) for (domain, language), counts in sorted(strata.items())
        ]
        for scope, position in (("language", 1), ("domain", 0)):
            grouped: dict[str, Counts] = {}
            for key, counts in sorted(strata.items()):
                grouped.setdefault(key[position], Counts()).combine(counts)
            scopes.extend(
                (scope, label if scope == "domain" else None, label if scope == "language" else None, counts)
                for label, counts in sorted(grouped.items())
            )
        aggregate = Counts()
        for counts in strata.values():
            aggregate.combine(counts)
        scopes.append(("aggregate", None, None, aggregate))
        for scope, domain, language, counts in scopes:
            records.append(
                {
                    "split": split,
                    "scope": scope,
                    "domain": domain,
                    "language": language,
                    "tokenizer": tokenizer.name,
                    "vocab_budget": len(tokenizer.vocab),
                    "status": "observed",
                    **counts.metrics(tokenizer.vocab, training_frequencies, rare_threshold),
                }
            )
    return records
