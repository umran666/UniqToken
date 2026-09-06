"""
Zero-Copy Memory-Mapped Binary Model Format (.uniqtok) for UniqToken (Issue #36).

Enables sub-millisecond cold start model loading via OS page mapping (mmap).
"""

from __future__ import annotations
import json
import mmap
import os
from pathlib import Path
import struct
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from uniqtoken.tokenizer import CustomTokenizer

# Binary Format Specifications
MAGIC = b"UNIQTOK\0"
FORMAT_VERSION = 1
# Header layout: magic(8s) + version, flags, vocab_size, max_subword_len (4I)
# + space_codepoint, scores_offset, offsets_offset, strings_data_offset,
# strings_data_len, config_json_offset, config_json_len (7Q) + reserved (12s).
HEADER_STRUCT = "<8sIIIIQQQQQQQ12s"
HEADER_SIZE = struct.calcsize(HEADER_STRUCT)


def export_binary(tokenizer: CustomTokenizer, output_path: Union[str, Path]) -> None:
    """Exports a CustomTokenizer into zero-copy binary format (.uniqtok)."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    model = tokenizer.model
    normalizer = tokenizer.normalizer
    pre_tok = tokenizer.pre_tokenizer
    # Sort tokens by ID 0..N-1
    token_to_id = model.token_to_id
    vocab = model.vocab
    if set(token_to_id.keys()) != set(vocab.keys()):
        raise ValueError(
            "binary export requires identical keys in model.vocab and model.token_to_id; use tokenizer.json instead"
        )
    vocab_size = len(token_to_id)
    sorted_tokens: List[Tuple[int, str, float]] = [(tid, tok, vocab[tok]) for tok, tid in token_to_id.items()]
    sorted_tokens.sort(key=lambda x: x[0])
    # The binary layout addresses tokens by position, so IDs must be dense.
    # Non-contiguous vocabularies (e.g. HF/SentencePiece imports with holes)
    # cannot round-trip exactly; refuse so callers fall back to tokenizer.json.
    if [tid for tid, _, _ in sorted_tokens] != list(range(vocab_size)):
        raise ValueError(
            "binary export requires contiguous token IDs 0..N-1 "
            f"(got {vocab_size} tokens with holes); use tokenizer.json instead"
        )
    # 1. Scores data (float32 array)
    scores_bytes = bytearray()
    for _, _, score in sorted_tokens:
        scores_bytes.extend(struct.pack("<f", score))
    # 2. String data & offsets
    string_data_bytes = bytearray()
    offsets_bytes = bytearray()
    current_offset = 0
    for _, tok, _ in sorted_tokens:
        tok_bytes = tok.encode("utf-8")
        tok_len = len(tok_bytes)
        offsets_bytes.extend(struct.pack("<II", current_offset, tok_len))
        string_data_bytes.extend(tok_bytes)
        current_offset += tok_len
    # 3. Config JSON
    config = {
        "special_tokens": model.special_tokens,
        "max_subword_len": model.max_subword_len,
        "byte_fallback": model.byte_fallback,
        "unk_token": model.unk_token,
        # Optional: absent (None) for tokenizers without the chat feature.
        "chat_template": getattr(tokenizer, "chat_template", None),
        "normalizer": {
            "space_char": normalizer.space_char,
            "lowercase": normalizer.lowercase,
            "casefold": normalizer.casefold,
            "normalize_unicode": normalizer.normalize_unicode,
            "normalize_punctuation": normalizer.normalize_punctuation,
            "normalize_unicode_spaces": normalizer.normalize_unicode_spaces,
            "collapse_whitespaces": normalizer.collapse_whitespaces,
            "strip_whitespace": normalizer.strip_whitespace,
        },
        "pre_tokenizer": {
            "space_char": pre_tok.space_char,
            "split_digits": pre_tok.split_digits,
            "split_punctuation": pre_tok.split_punctuation,
            "keep_special_tokens": pre_tok.keep_special_tokens,
            "special_token_pattern": pre_tok.special_token_pattern,
            "hex_literals": pre_tok.hex_literals,
            "digit_chunk_size": pre_tok.digit_chunk_size,
            "digit_chunking": pre_tok.digit_chunking,
            "preset": pre_tok.preset,
        },
    }
    config_bytes = json.dumps(config, ensure_ascii=False).encode("utf-8")
    # Compute section offsets
    scores_offset = HEADER_SIZE
    offsets_offset = scores_offset + len(scores_bytes)
    strings_data_offset = offsets_offset + len(offsets_bytes)
    config_json_offset = strings_data_offset + len(string_data_bytes)
    flags = 1 if model.byte_fallback else 0
    space_codepoint = ord(normalizer.space_char) if normalizer.space_char else 0x2581
    header = struct.pack(
        HEADER_STRUCT,
        MAGIC,
        FORMAT_VERSION,
        flags,
        vocab_size,
        model.max_subword_len,
        space_codepoint,
        scores_offset,
        offsets_offset,
        strings_data_offset,
        len(string_data_bytes),
        config_json_offset,
        len(config_bytes),
        b"\0" * 12,
    )
    with tempfile.NamedTemporaryFile(
        dir=out_file.parent,
        prefix=f"{out_file.name}.tmp.",
        delete=False,
    ) as f:
        tmp_file = Path(f.name)
        try:
            f.write(header)
            f.write(scores_bytes)
            f.write(offsets_bytes)
            f.write(string_data_bytes)
            f.write(config_bytes)
        except (OSError, ValueError):
            if tmp_file.is_file():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
            raise
    try:
        tmp_file.replace(out_file)
    except (OSError, ValueError):
        if tmp_file.is_file():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        raise


def load_binary(file_path: Union[str, Path], use_mmap: bool = True) -> CustomTokenizer:
    """Loads a CustomTokenizer from binary format (.uniqtok) using zero-copy mmap."""
    from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
    from uniqtoken.tokenizer import CustomTokenizer
    from uniqtoken.unigram_trainer import UnigramModel

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Binary model file not found: {path}")
    f = open(path, "rb")  # noqa: SIM115
    try:
        mm: Union[mmap.mmap, bytes]
        if use_mmap:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            mm = f.read()
        if len(mm) < HEADER_SIZE:
            raise ValueError(f"Corrupted binary model: file size ({len(mm)}) < header size")
        (
            magic,
            version,
            flags,
            vocab_size,
            max_subword_len,
            space_codepoint,
            scores_offset,
            offsets_offset,
            strings_data_offset,
            strings_data_len,
            config_json_offset,
            config_json_len,
            _,
        ) = struct.unpack(HEADER_STRUCT, mm[:HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(f"Invalid magic bytes in {path}: expected {MAGIC!r}, got {magic!r}")
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported binary format version: {version}")
        if not (0 <= space_codepoint <= 0x10FFFF) or (0xD800 <= space_codepoint <= 0xDFFF):
            raise ValueError(f"Corrupted binary model: invalid space codepoint {space_codepoint:#x}")
        file_size = len(mm)
        scores_end = scores_offset + vocab_size * 4
        offsets_end = offsets_offset + vocab_size * 8
        strings_end = strings_data_offset + strings_data_len
        config_end = config_json_offset + config_json_len
        if not (
            HEADER_SIZE
            <= scores_offset
            <= scores_end
            <= offsets_offset
            <= offsets_end
            <= strings_data_offset
            <= strings_end
            <= config_json_offset
            <= config_end
            <= file_size
        ):
            raise ValueError("Corrupted binary model: invalid section offsets")
        # Parse config JSON
        config_data = bytes(mm[config_json_offset : config_json_offset + config_json_len])
        try:
            config = json.loads(config_data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            raise ValueError(f"Corrupted binary model: invalid config JSON: {e}") from e
        if not isinstance(config, dict):
            raise ValueError("Corrupted binary model: config must be a JSON object")
        norm_cfg = config.get("normalizer", {})
        if not isinstance(norm_cfg, dict):
            raise ValueError("Corrupted binary model: normalizer config must be a JSON object")
        for key in (
            "lowercase",
            "casefold",
            "normalize_unicode",
            "normalize_punctuation",
            "normalize_unicode_spaces",
            "collapse_whitespaces",
            "strip_whitespace",
        ):
            if key in norm_cfg and not isinstance(norm_cfg[key], bool):
                raise ValueError(f"Corrupted binary model: normalizer.{key} must be a boolean")
        if "space_char" in norm_cfg and not isinstance(norm_cfg["space_char"], str):
            raise ValueError("Corrupted binary model: normalizer.space_char must be a string")
        pre_cfg = config.get("pre_tokenizer", {})
        if not isinstance(pre_cfg, dict):
            raise ValueError("Corrupted binary model: pre_tokenizer config must be a JSON object")
        for key in (
            "split_digits",
            "split_punctuation",
            "keep_special_tokens",
            "hex_literals",
        ):
            if key in pre_cfg and not isinstance(pre_cfg[key], bool):
                raise ValueError(f"Corrupted binary model: pre_tokenizer.{key} must be a boolean")
        if "space_char" in pre_cfg and not isinstance(pre_cfg["space_char"], str):
            raise ValueError("Corrupted binary model: pre_tokenizer.space_char must be a string")
        if "special_token_pattern" in pre_cfg and not isinstance(pre_cfg["special_token_pattern"], str):
            raise ValueError("Corrupted binary model: pre_tokenizer.special_token_pattern must be a string")
        if (
            "digit_chunk_size" in pre_cfg
            and pre_cfg["digit_chunk_size"] is not None
            and (isinstance(pre_cfg["digit_chunk_size"], bool) or not isinstance(pre_cfg["digit_chunk_size"], int))
        ):
            raise ValueError("Corrupted binary model: pre_tokenizer.digit_chunk_size must be an int or None")
        if "digit_chunking" in pre_cfg and pre_cfg["digit_chunking"] not in ("block3", "single", "greedy"):
            raise ValueError(
                f"Corrupted binary model: invalid pre_tokenizer.digit_chunking: {pre_cfg['digit_chunking']!r}"
            )
        if "preset" in pre_cfg and pre_cfg["preset"] is not None and not isinstance(pre_cfg["preset"], str):
            raise ValueError("Corrupted binary model: pre_tokenizer.preset must be a string or None")
        if "special_tokens" in config and not isinstance(config["special_tokens"], list):
            raise ValueError("Corrupted binary model: special_tokens must be a list")
        if "unk_token" in config and not isinstance(config["unk_token"], str):
            raise ValueError("Corrupted binary model: unk_token must be a string")
        if (
            "chat_template" in config
            and config["chat_template"] is not None
            and not isinstance(config["chat_template"], str)
        ):
            raise ValueError("Corrupted binary model: chat_template must be a string or None")
        # Fast unpack scores and strings
        vocab: Dict[str, float] = {}
        token_to_id: Dict[str, int] = {}
        id_to_token: Dict[int, str] = {}
        scores_fmt = f"<{vocab_size}f"
        scores = struct.unpack_from(scores_fmt, mm, scores_offset)
        offsets_fmt = f"<{vocab_size * 2}I"
        offsets_flat = struct.unpack_from(offsets_fmt, mm, offsets_offset)
        strings_base = strings_data_offset
        mv = memoryview(mm)
        for i in range(vocab_size):
            off = offsets_flat[i * 2]
            length = offsets_flat[i * 2 + 1]
            if off + length > strings_data_len:
                raise ValueError("Corrupted binary model: invalid token offset")
            tok = str(mv[strings_base + off : strings_base + off + length], "utf-8")
            if tok in vocab:
                raise ValueError(f"Corrupted binary model: duplicate token {tok!r}")
            score = scores[i]
            vocab[tok] = score
            token_to_id[tok] = i
            id_to_token[i] = tok
        byte_fallback = bool(flags & 1)
        try:
            model = UnigramModel(
                vocab=vocab,
                token_to_id=token_to_id,
                id_to_token=id_to_token,
                special_tokens=config.get("special_tokens", []),
                max_subword_len=max_subword_len,
                byte_fallback=byte_fallback,
                unk_token=config.get("unk_token", "<|unk|>"),
            )
            normalizer = Normalizer(
                space_char=norm_cfg.get("space_char", chr(space_codepoint)),
                lowercase=norm_cfg.get("lowercase", False),
                casefold=norm_cfg.get("casefold", False),
                normalize_unicode=norm_cfg.get("normalize_unicode", True),
                normalize_punctuation=norm_cfg.get("normalize_punctuation", False),
                normalize_unicode_spaces=norm_cfg.get("normalize_unicode_spaces", True),
                collapse_whitespaces=norm_cfg.get("collapse_whitespaces", False),
                strip_whitespace=norm_cfg.get("strip_whitespace", False),
            )
            pre_tokenizer = RegexPreTokenizer(
                space_char=pre_cfg.get("space_char", chr(space_codepoint)),
                split_digits=pre_cfg.get("split_digits", False),
                split_punctuation=pre_cfg.get("split_punctuation", True),
                keep_special_tokens=pre_cfg.get("keep_special_tokens", True),
                special_token_pattern=pre_cfg.get("special_token_pattern", r"<\|[^\s|]+\|>"),
                hex_literals=pre_cfg.get("hex_literals", True),
                digit_chunk_size=pre_cfg.get("digit_chunk_size"),
                digit_chunking=pre_cfg.get("digit_chunking", "greedy"),
                preset=pre_cfg.get("preset"),
            )
            tokenizer = CustomTokenizer(
                model=model,
                normalizer=normalizer,
                pre_tokenizer=pre_tokenizer,
            )
            tokenizer.chat_template = config.get("chat_template", None)
            return tokenizer
        except (TypeError, ValueError, KeyError) as e:
            raise ValueError(f"Corrupted binary model: invalid component configuration: {e}") from e
    finally:
        f.close()
