from __future__ import annotations

import base64
import re as _stdlib_re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .bpe_model import BPEModel

try:
    # Exact parity with tiktoken's Rust regex needs \p{L}/\p{N} classes and
    # possessive quantifiers (?+ / ++) — only the third-party `regex` module
    # supports both on Python.
    import regex as _re
except ImportError:  # pragma: no cover
    _re = None


#: Pre-tokenization patterns, verbatim from tiktoken's encodings.
TIKTOKEN_PATTERNS: Dict[str, str] = {
    "gpt2": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""",
    "cl100k_base": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
    "o200k_base": (
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
        r"""|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
        r"""|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    ),
}


def load_tiktoken_ranks(path: Union[str, Path]) -> Dict[bytes, int]:
    """
    Loads a tiktoken ``.tiktoken`` ranks file (``<base64(token bytes)> <rank>`` per line).

    Returns a mapping from token byte strings to merge ranks. All 256 single
    bytes must be present, as tiktoken's algorithm requires full byte coverage.
    """
    ranks: Dict[bytes, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                token_b64, rank_str = line.split()
                token_bytes = base64.b64decode(token_b64, validate=True)
                rank = int(rank_str)
            except ValueError as exc:
                raise ValueError(f"malformed tiktoken ranks line: {line[:80]!r}") from exc
            if rank < 0:
                raise ValueError(f"tiktoken rank must be non-negative, got {rank}")
            if token_bytes in ranks:
                raise ValueError(f"duplicate rank entry for {token_bytes!r}")
            ranks[token_bytes] = rank
    single_bytes = {token[0] for token in ranks if len(token) == 1}
    if single_bytes != set(range(256)):
        raise ValueError(
            "tiktoken ranks file must contain exactly one rank entry for every "
            f"single byte 0x00-0xFF (found {len(single_bytes)}/256)"
        )
    return ranks


class TiktokenEncoding:
    """
    Tiktoken-compatible byte-level BPE encoding loaded from a ranks file.

    Produces the SAME integer IDs as tiktoken for the same vocabulary, without
    requiring the tiktoken package: regex pre-tokenization (tiktoken's exact
    patterns) followed by greedy lowest-rank byte-pair merging per piece.

    Requires the ``regex`` package for the bundled patterns. A raw ``pattern``
    string may be supplied instead of a preset name.
    """

    def __init__(
        self,
        name: str,
        ranks: Dict[bytes, int],
        pattern: str,
        special_tokens: Optional[Dict[str, int]] = None,
        explicit_n_vocab: Optional[int] = None,
    ):
        if _re is None:
            raise ImportError(
                "the 'regex' package is required for tiktoken pattern compatibility; install it with: pip install regex"
            )
        self.name = name
        self.ranks = dict(ranks)
        self.pattern = pattern
        self._compiled = _re.compile(pattern)
        self.special_tokens = dict(special_tokens or {})
        self._id_to_special = {v: k for k, v in self.special_tokens.items()}
        self._rank_to_bytes = {r: b for b, r in self.ranks.items()}
        if len(self._rank_to_bytes) != len(self.ranks):
            raise ValueError("duplicate ranks in vocabulary")
        self.n_vocab = (
            explicit_n_vocab
            or max(max(self.ranks.values(), default=-1), max(self.special_tokens.values(), default=-1)) + 1
        )

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        name: str = "custom",
        pattern: str = "cl100k_base",
        special_tokens: Optional[Dict[str, int]] = None,
        explicit_n_vocab: Optional[int] = None,
    ) -> "TiktokenEncoding":
        """
        Loads a ``.tiktoken`` ranks file. ``pattern`` is one of TIKTOKEN_PATTERNS
        preset names ("gpt2", "cl100k_base", "o200k_base") or a raw regex string.
        """
        resolved = TIKTOKEN_PATTERNS.get(pattern, pattern)
        return cls(
            name=name,
            ranks=load_tiktoken_ranks(path),
            pattern=resolved,
            special_tokens=special_tokens,
            explicit_n_vocab=explicit_n_vocab,
        )

    @property
    def vocab_size(self) -> int:
        return self.n_vocab

    # -- core byte-level BPE (tiktoken algorithm) ---------------------------

    def _merge_piece(self, piece: bytes) -> List[bytes]:
        # ponytail: O(n^2) worst case like tiktoken's educational impl; the
        # linked-parts heap version is an optimization, not a semantic change.
        parts: List[bytes] = [piece[i : i + 1] for i in range(len(piece))]
        while len(parts) > 1:
            best_rank: Optional[int] = None
            best_idx = -1
            for i in range(len(parts) - 1):
                rank = self.ranks.get(parts[i] + parts[i + 1])
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            parts[best_idx : best_idx + 2] = [parts[best_idx] + parts[best_idx + 1]]
        return parts

    def _encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        for match in self._compiled.finditer(text):
            piece = match.group(0).encode("utf-8")
            rank = self.ranks.get(piece)
            if rank is not None:
                ids.append(rank)
                continue
            for token in self._merge_piece(piece):
                token_rank = self.ranks.get(token)
                if token_rank is None:
                    raise ValueError(f"no rank for byte token {token!r} in {self.name}")
                ids.append(token_rank)
        return ids

    # -- public API ----------------------------------------------------------

    def encode(self, text: str, allowed_special: Union[str, set] = "none") -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if allowed_special == "all":
            allowed = set(self.special_tokens)
        elif allowed_special == "none" or not allowed_special:
            allowed = set()
        else:
            allowed = set(allowed_special)

        if self.special_tokens:
            re_engine = _re if _re is not None else _stdlib_re
            # Build pattern from all special tokens longest-first to prevent prefix collisions
            all_special_pattern = (
                "(" + "|".join(re_engine.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            )
            for match in re_engine.finditer(all_special_pattern, text):
                if match.group(0) not in allowed:
                    raise ValueError(f"Encountered text corresponding to disallowed special token {match.group(0)!r}.")

        if not allowed:
            return self._encode_ordinary(text)

        # Split on allowed special tokens (longest-first alternation), map each
        # occurrence to its ID, and byte-encode everything between.
        special_pattern = _stdlib_re.compile(
            "(" + "|".join(_stdlib_re.escape(s) for s in sorted(allowed, key=len, reverse=True)) + ")"
        )
        ids: List[int] = []
        for segment in special_pattern.split(text):
            if segment in allowed:
                ids.append(self.special_tokens[segment])
            elif segment:
                ids.extend(self._encode_ordinary(segment))
        return ids

    def encode_to_ids(self, text: str, allowed_special: Union[str, set] = "none") -> List[int]:
        return self.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids: List[int]) -> str:
        if not isinstance(token_ids, list):
            raise TypeError(f"token_ids must be a list of ints, got {type(token_ids).__name__}")
        pieces: List[bytes] = []
        for tid in token_ids:
            if tid in self._rank_to_bytes:
                pieces.append(self._rank_to_bytes[tid])
            elif tid in self._id_to_special:
                pieces.append(self._id_to_special[tid].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {tid} in {self.name}")
        return b"".join(pieces).decode("utf-8", errors="replace")

    # -- conversion to UniqToken's native model --------------------------------

    def to_uniqtoken_bpe_model(self) -> BPEModel:
        """
        Converts the tiktoken ranks into a UniqToken :class:`BPEModel`.

        Byte strings map 1:1 to str via latin-1 (every rank key is a byte
        string, so the mapping is bijective and round-trips exactly). Merge
        pairs are reconstructed by re-segmenting each multi-byte token under
        strictly lower ranks — the segmentation BPE training produced when the
        token was created. Token IDs (ranks) are preserved exactly.

        Note: the returned model's vocab/merges/IDs are faithful, but UniqToken's
        BPEModel pre-tokenizes on spaces only — for tiktoken-identical output,
        keep using TiktokenEncoding.encode.
        """
        vocab: set = {b.decode("latin-1") for b in self.ranks}
        token_to_id: Dict[str, int] = {b.decode("latin-1"): r for b, r in self.ranks.items()}
        id_to_token: Dict[int, str] = {r: b.decode("latin-1") for b, r in self.ranks.items()}
        for special, sid in self.special_tokens.items():
            token_to_id[special] = sid
            id_to_token[sid] = special
            vocab.add(special)

        merges: Dict[Tuple[str, str], int] = {}
        lower_ranks: Dict[bytes, int] = {}
        for token_bytes, rank in sorted(self.ranks.items(), key=lambda kv: kv[1]):
            if len(token_bytes) < 2:
                lower_ranks[token_bytes] = rank
                continue
            seg = self._merge_piece_with_ranks(token_bytes, lower_ranks)
            if len(seg) == 2:
                merges[(seg[0].decode("latin-1"), seg[1].decode("latin-1"))] = rank
            else:
                # Not derivable as a two-symbol merge under lower ranks (can
                # happen with hand-crafted ranks) — fall back to the known-token
                # split with the lowest combined part ranks.
                best: Optional[Tuple[int, bytes, bytes]] = None
                for i in range(1, len(token_bytes)):
                    left, right = token_bytes[:i], token_bytes[i:]
                    if left in self.ranks and right in self.ranks:
                        score = self.ranks[left] + self.ranks[right]
                        if best is None or score < best[0]:
                            best = (score, left, right)
                if best is not None:
                    merges[(best[1].decode("latin-1"), best[2].decode("latin-1"))] = rank
            lower_ranks[token_bytes] = rank

        return BPEModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            merges=merges,
            special_tokens=list(self.special_tokens),
            byte_fallback=False,  # ranks already cover every single byte
        )

    def _merge_piece_with_ranks(self, piece: bytes, ranks: Dict[bytes, int]) -> List[bytes]:
        parts: List[bytes] = [piece[i : i + 1] for i in range(len(piece))]
        while len(parts) > 1:
            best_rank: Optional[int] = None
            best_idx = -1
            for i in range(len(parts) - 1):
                rank = ranks.get(parts[i] + parts[i + 1])
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            parts[best_idx : best_idx + 2] = [parts[best_idx] + parts[best_idx + 1]]
        return parts
