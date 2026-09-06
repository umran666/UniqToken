from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .bpe_model import BPEModel
from .pre_tokenizer import Normalizer, RegexPreTokenizer
from .tokenizer import CustomTokenizer
from .unigram_trainer import UnigramModel

try:
    # ByteLevel/GPT-2 pre-tokenization needs \p{L}/\p{N}; same requirement as
    # the tiktoken adapter.
    import regex as _re
    from .tiktoken_adapter import TIKTOKEN_PATTERNS
except ImportError:  # pragma: no cover
    _re = None
    TIKTOKEN_PATTERNS = {}

DEFAULT_SPECIAL_PATTERN = r"<\|[^\s|]+\|>"


def _warn_unsupported(component: str, detail: str) -> None:
    warnings.warn(
        f"HF importer: {component} ({detail}) has no exact UniqToken equivalent; "
        "imported tokenizer may tokenize differently than the source.",
        stacklevel=3,
    )


def _map_normalizer(cfg: Any) -> Normalizer:
    """Maps an HF normalizer config onto UniqToken's Normalizer (best effort)."""
    cfg = cfg or {}
    ntype = cfg.get("type")
    # An absent/empty HF normalizer is an identity transform. Do not let
    # UniqToken's Normalizer defaults silently add NFKC or Unicode-space mapping.
    kwargs: Dict[str, Any] = {
        "space_char": "\u2581",
        "normalize_unicode": False,
        "normalize_unicode_spaces": False,
    }

    def walk(node: Any) -> None:
        if not node:
            return
        if node.get("type") == "Sequence":
            for child in node.get("normalizers", []):
                walk(child)
            return
        t = node.get("type")
        if t == "NFKC":
            kwargs["normalize_unicode"] = True
        elif t == "Lowercase":
            kwargs["lowercase"] = True
        elif t in ("NFC", "NFD", "NFKD", "StripAccents", "Replace", "Strip", "Prepend", "ByteLevel"):
            _warn_unsupported("normalizer", t)
        else:
            _warn_unsupported("normalizer", t or "unknown")

    walk(cfg)
    return Normalizer(**kwargs)


def _map_pre_tokenizer(cfg: Any, normalizer: Normalizer) -> RegexPreTokenizer:
    """Maps an HF pre-tokenizer config onto UniqToken's RegexPreTokenizer (best effort)."""
    cfg = cfg or {}
    ptype = cfg.get("type")
    space_char = "\u2581"
    if ptype == "Metaspace":
        space_char = cfg.get("replacement", "\u2581") or "\u2581"
        if not isinstance(space_char, str) or len(space_char) != 1:
            raise ValueError("HF Metaspace replacement must be exactly one character")
        if space_char in {Normalizer._ESCAPE_PREFIX, Normalizer._ESCAPED_METASPACE}:
            raise ValueError("HF Metaspace replacement conflicts with UniqToken's reserved escape characters")
        normalizer.space_char = space_char
        prepend = cfg.get("prepend_scheme") or ("always" if cfg.get("add_prefix_space") else "never")
        if prepend != "never":
            _warn_unsupported("pre_tokenizer", f"Metaspace prepend_scheme={prepend!r} (UniqToken never prepends)")
        if cfg.get("split") is False:
            _warn_unsupported("pre_tokenizer", "Metaspace split=False")
    elif ptype not in (None, "Metaspace"):
        _warn_unsupported("pre_tokenizer", ptype or "unknown")
    elif ptype is None:
        _warn_unsupported("pre_tokenizer", "none configured; UniqToken's default regex will be used")

    return RegexPreTokenizer(space_char=space_char)


def import_hf_unigram(data: Dict[str, Any]) -> CustomTokenizer:
    """
    Imports an HF ``tokenizer.json`` (parsed dict) with a Unigram model into a
    UniqToken :class:`CustomTokenizer`.

    Vocab scores and token IDs are preserved exactly. Normalizer/pre-tokenizer
    components are mapped best-effort; anything without an exact UniqToken
    equivalent emits a warning. Requires ``byte_fallback`` or an unknown token
    for OOV characters.
    """
    model = data.get("model", {})
    if model.get("type") != "Unigram":
        raise ValueError(f"expected a Unigram model, got {model.get('type')!r}")

    vocab_list = model.get("vocab", [])
    if not vocab_list:
        raise ValueError("HF Unigram model has an empty vocab")

    vocab: Dict[str, float] = {}
    token_to_id: Dict[str, int] = {}
    id_to_token: Dict[int, str] = {}
    for idx, entry in enumerate(vocab_list):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError(f"malformed HF Unigram vocab entry at index {idx}")
        token = entry[0]
        if not isinstance(token, str) or not token:
            raise ValueError(f"HF Unigram vocab token at index {idx} must be a non-empty string")
        try:
            score = float(entry[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"HF Unigram score at index {idx} is not numeric") from exc
        if not math.isfinite(score):
            raise ValueError(f"HF Unigram score at index {idx} must be finite")
        if token in token_to_id:
            raise ValueError(f"HF Unigram vocab contains duplicate token {token!r}")
        vocab[token] = score
        token_to_id[token] = idx
        id_to_token[idx] = token

    special_tokens: List[str] = []
    for added in data.get("added_tokens", []):
        token = added["content"]
        raw_id = added["id"]
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            raise ValueError("HF added token IDs must be integers")
        added_id = raw_id
        if not isinstance(token, str) or not token:
            raise ValueError("HF added token content must be a non-empty string")
        if added_id < 0:
            raise ValueError("HF added token IDs must be non-negative")
        if added.get("special", False):
            special_tokens.append(token)
        if token not in token_to_id:
            if added_id in id_to_token:
                raise ValueError(
                    f"HF added token {token!r} reuses ID {added_id} already assigned to {id_to_token[added_id]!r}"
                )
            token_to_id[token] = added_id
            id_to_token[added_id] = token
            vocab.setdefault(token, -10.0)  # HF Unigram needs a score for every id
        elif token_to_id[token] != added_id:
            raise ValueError(f"HF added token {token!r} has conflicting IDs {token_to_id[token]} and {added_id}")

    byte_fallback = bool(model.get("byte_fallback", False))
    unk_id = model.get("unk_id")
    if unk_id is not None and (
        not isinstance(unk_id, int) or isinstance(unk_id, bool) or unk_id < 0 or unk_id not in id_to_token
    ):
        raise ValueError("HF Unigram unk_id must reference a non-negative integer vocabulary ID")
    unk_token = id_to_token.get(unk_id) if unk_id is not None else None
    if not byte_fallback and unk_token is None:
        raise ValueError(
            "HF Unigram import requires byte_fallback=true or an UNKNOWN token; the model contains neither"
        )

    normalizer = _map_normalizer(data.get("normalizer"))
    pre_tokenizer = _map_pre_tokenizer(data.get("pre_tokenizer"), normalizer)

    unigram = UnigramModel(
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        special_tokens=special_tokens,
        max_subword_len=max(max(len(t) for t in vocab), 1),
        byte_fallback=byte_fallback,
        unk_token=unk_token or "<|unk|>",
    )
    return CustomTokenizer(normalizer=normalizer, pre_tokenizer=pre_tokenizer, model=unigram)


def _bytes_to_unicode() -> Dict[int, str]:
    """Standard GPT-2 byte-to-unicode table (every byte -> printable char)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class HFByteLevelBPE:
    """
    Byte-level BPE encoding loaded from an HF ``tokenizer.json`` (GPT-2 style).

    Produces the SAME integer IDs as the equivalent HuggingFace ``tokenizers``
    Tokenizer (ByteLevel pre-tokenizer + BPE model + ByteLevel decoder) using
    tiktoken's GPT-2 split pattern. Requires the ``regex`` package.
    """

    def __init__(
        self,
        name: str,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        special_tokens: Optional[Dict[str, int]] = None,
        add_prefix_space: bool = False,
        pattern: Optional[str] = None,
    ):
        if _re is None:
            raise ImportError("the 'regex' package is required for ByteLevel BPE import")
        self.name = name
        self.vocab = dict(vocab)
        if any(not isinstance(token, str) for token in self.vocab):
            raise ValueError("HF BPE vocab tokens must be strings")
        if any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in self.vocab.values()
        ):
            raise ValueError("HF BPE vocab IDs must be non-negative integers")
        if len(set(self.vocab.values())) != len(self.vocab):
            raise ValueError("HF BPE vocab IDs must be unique")
        if any(
            not isinstance(pair, tuple) or len(pair) != 2 or not all(isinstance(part, str) for part in pair)
            for pair in merges
        ):
            raise ValueError("HF BPE merge entries must be pairs of strings")
        if len(set(merges)) != len(merges):
            raise ValueError("HF BPE merge entries must be unique")
        self.ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = dict(special_tokens or {})
        if any(not isinstance(token, str) for token in self.special_tokens):
            raise ValueError("HF special-token contents must be strings")
        if any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in self.special_tokens.values()
        ):
            raise ValueError("HF special-token IDs must be non-negative integers")
        if len(set(self.special_tokens.values())) != len(self.special_tokens):
            raise ValueError("HF special-token IDs must be unique")
        for token, token_id in self.special_tokens.items():
            owner = next((name for name, value in self.vocab.items() if value == token_id), None)
            if owner is not None and owner != token:
                raise ValueError(f"HF special token {token!r} reuses ID {token_id} assigned to {owner!r}")
        self._id_to_special = {v: k for k, v in self.special_tokens.items()}
        self._id_to_token = {i: t for t, i in self.vocab.items()}
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}
        self.add_prefix_space = add_prefix_space
        if pattern in TIKTOKEN_PATTERNS:
            self._split_re = _re.compile(TIKTOKEN_PATTERNS[pattern])
        elif pattern is not None:
            self._split_re = _re.compile(pattern)
        else:
            self._split_re = _re.compile(TIKTOKEN_PATTERNS["gpt2"])
        self.n_vocab = max(self.vocab.values(), default=-1) + 1

    @property
    def vocab_size(self) -> int:
        return self.n_vocab

    def _bpe(self, piece: str) -> List[str]:
        # ponytail: O(n^2) greedy lowest-rank merge — same result as HF's BPE,
        # heap version is an optimization not a semantic change.
        parts: List[str] = list(piece)
        while len(parts) > 1:
            best_rank: Optional[int] = None
            best_idx = -1
            for i in range(len(parts) - 1):
                rank = self.ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            parts[best_idx : best_idx + 2] = [parts[best_idx] + parts[best_idx + 1]]
        return parts

    def _encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        if self.add_prefix_space:
            text = " " + text
        for piece in self._split_re.findall(text):
            mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for token in self._bpe(mapped):
                tid = self.vocab.get(token)
                if tid is None:
                    raise ValueError(f"no vocab id for token {token!r} in {self.name}")
                ids.append(tid)
        return ids

    def encode(self, text: str, allowed_special: Union[str, set] = "none") -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if allowed_special == "all":
            allowed = set(self.special_tokens)
        elif allowed_special == "none" or not allowed_special:
            allowed = set()
        else:
            allowed = set(allowed_special)

        disallowed = set(self.special_tokens) - allowed
        if disallowed:
            import re as _stdlib_fallback

            re_engine = _re if _re is not None else _stdlib_fallback
            disallowed_pattern = (
                "(" + "|".join(re_engine.escape(s) for s in sorted(disallowed, key=len, reverse=True)) + ")"
            )
            match = re_engine.search(disallowed_pattern, text)
            if match:
                raise ValueError(f"Encountered text corresponding to disallowed special token {match.group(0)!r}.")

        if not allowed:
            return self._encode_ordinary(text)

        import re as _stdlib_re

        special_re = _stdlib_re.compile(
            "(" + "|".join(_stdlib_re.escape(s) for s in sorted(allowed, key=len, reverse=True)) + ")"
        )
        ids: List[int] = []
        for segment in special_re.split(text):
            if segment in allowed:
                ids.append(self.special_tokens[segment])
            elif segment:
                ids.extend(self._encode_ordinary(segment))
        return ids

    def decode(self, token_ids: List[int]) -> str:
        pieces: List[bytes] = []
        for tid in token_ids:
            if tid in self._id_to_special:
                pieces.append(self._id_to_special[tid].encode("utf-8"))
                continue
            token = self._id_to_token.get(tid)
            if token is None:
                raise ValueError(f"unknown token id {tid} in {self.name}")
            pieces.append(bytes(self.byte_decoder[c] for c in token))
        return b"".join(pieces).decode("utf-8", errors="replace")


def import_hf_bpe(data: Dict[str, Any]) -> Union[HFByteLevelBPE, BPEModel]:
    """
    Imports an HF ``tokenizer.json`` BPE model.

    ByteLevel pre-tokenization (GPT-2 style) returns a fully functional
    :class:`HFByteLevelBPE` with exact-ID encode/decode. Non-byte-level BPE
    returns a UniqToken :class:`BPEModel` carrying vocab/merges/IDs for data
    reuse — encode semantics depend on the source pre-tokenizer, which UniqToken
    cannot reproduce in general.
    """
    model = data.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"expected a BPE model, got {model.get('type')!r}")

    vocab: Dict[str, int] = dict(model.get("vocab", {}))
    if not vocab:
        raise ValueError("HF BPE model has an empty vocab")
    merges: List[Tuple[str, str]] = []
    for entry in model.get("merges", []):
        if isinstance(entry, str):
            if " " not in entry:
                raise ValueError(f"malformed merge entry: {entry!r}")
            left, right = entry.split(" ", 1)
            merges.append((left, right))
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            merges.append((entry[0], entry[1]))
        else:
            raise ValueError(f"malformed merge entry: {entry!r}")

    special_tokens: Dict[str, int] = {}
    for added in data.get("added_tokens", []):
        content = added.get("content")
        if not added.get("special"):
            raise ValueError(f"HF BPE added token {content!r} is not special and cannot be represented exactly")
        added_id = added.get("id")
        if not isinstance(added_id, int) or isinstance(added_id, bool) or added_id < 0:
            raise ValueError("HF special-token IDs must be non-negative integers")
        if content in special_tokens and special_tokens[content] != added_id:
            raise ValueError(f"HF special token {content!r} has conflicting IDs")
        special_tokens[content] = added_id
    pt = data.get("pre_tokenizer") or {}
    pt_type = pt.get("type")
    byte_cfg: Dict[str, Any] = {}
    pattern_str: Optional[str] = None
    if pt_type == "Sequence":
        for child in pt.get("pretokenizers", []):
            ctype = child.get("type")
            if ctype == "ByteLevel":
                byte_cfg = child
                pt_type = "ByteLevel"
            elif ctype == "Split":
                pinfo = child.get("pattern", {})
                if isinstance(pinfo, dict) and "Regex" in pinfo:
                    pattern_str = pinfo["Regex"]

    if pt_type == "ByteLevel":
        return HFByteLevelBPE(
            name="hf_bpe",
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
            add_prefix_space=bool(byte_cfg.get("add_prefix_space", False)),
            pattern=pattern_str,
        )

    _warn_unsupported("pre_tokenizer", f"{pt_type!r} BPE import returns vocab/merges only")
    id_to_token = {i: t for t, i in vocab.items()}
    return BPEModel(
        vocab=set(vocab),
        token_to_id=vocab,
        id_to_token=id_to_token,
        merges={pair: i for i, pair in enumerate(merges)},
        special_tokens=list(special_tokens),
        byte_fallback=False,
    )


def import_hf_tokenizer(source: Union[str, Path, Dict[str, Any]]) -> Union[CustomTokenizer, HFByteLevelBPE, BPEModel]:
    """
    Imports an HF ``tokenizer.json`` file (or parsed dict), dispatching on the
    model type: Unigram -> CustomTokenizer, BPE -> HFByteLevelBPE/BPEModel.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            path = path / "tokenizer.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source

    mtype = data.get("model", {}).get("type")
    if mtype == "Unigram":
        return import_hf_unigram(data)
    if mtype == "BPE":
        return import_hf_bpe(data)
    raise NotImplementedError(
        f"HF model type {mtype!r} is not supported (UniqToken has no WordPiece engine); supported types: Unigram, BPE"
    )
