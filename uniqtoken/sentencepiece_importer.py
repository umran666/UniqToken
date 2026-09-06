from __future__ import annotations

import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .pre_tokenizer import Normalizer, RegexPreTokenizer
from .tokenizer import CustomTokenizer
from .unigram_trainer import UnigramModel

# SentencePiece PieceType enum (sentencepiece_model.proto)
SP_NORMAL = 1
SP_UNKNOWN = 2
SP_CONTROL = 3
SP_USER_DEFINED = 4
SP_UNUSED = 5
SP_BYTE = 6

# TrainerSpec.model_type enum
SP_MODEL_UNIGRAM = 1
SP_MODEL_BPE = 2
SP_MODEL_WORD = 3
SP_MODEL_CHAR = 4

SP_MODEL_TYPE_NAMES = {1: "unigram", 2: "bpe", 3: "word", 4: "char"}


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    for index in range(10):
        if pos >= len(buf):
            raise ValueError("truncated varint in SentencePiece model")
        byte = buf[pos]
        pos += 1
        if index == 9 and byte > 1:
            raise ValueError("varint exceeds the 64-bit protobuf limit")
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
    raise ValueError("varint exceeds the 64-bit protobuf limit")


def _iter_fields(buf: bytes):
    """Yields (field_number, wire_type, value) over a protobuf buffer."""
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = _read_varint(buf, pos)
        field_number, wire_type = key >> 3, key & 0x07
        if field_number == 0:
            raise ValueError("protobuf field number must be positive")
        value: Union[int, bytes]
        if wire_type == 0:
            value, pos = _read_varint(buf, pos)
        elif wire_type == 1:
            if pos + 8 > end:
                raise ValueError("truncated fixed64 field in SentencePiece model")
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos)
            if length > end - pos:
                raise ValueError("truncated length-delimited field in SentencePiece model")
            value, pos = buf[pos : pos + length], pos + length
        elif wire_type == 5:
            if pos + 4 > end:
                raise ValueError("truncated fixed32 field in SentencePiece model")
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type} in SentencePiece model")
        yield field_number, wire_type, value


@dataclass
class SentencePieceModel:
    """Parsed subset of SentencePiece's ModelProto (zero protobuf dependency)."""

    pieces: List[Tuple[str, float, int]] = field(default_factory=list)  # (piece, score, type)
    model_type: int = SP_MODEL_UNIGRAM
    normalizer_name: str = ""
    add_dummy_prefix: bool = True


def parse_sentencepiece_proto(data: bytes) -> SentencePieceModel:
    """
    Parses a serialized SentencePiece ``.model`` protobuf (raw wire format —
    no protobuf dependency). Extracts the vocabulary pieces with scores and
    types, the trainer model type, and the normalizer name/dummy-prefix flag.
    """
    model = SentencePieceModel()
    for field_number, wire_type, value in _iter_fields(data):
        if field_number == 1 and wire_type == 2:  # repeated SentencePiece pieces
            piece: str = ""
            score: float = 0.0
            ptype: int = SP_NORMAL
            for f, w, v in _iter_fields(value):
                if f == 1 and w == 2:
                    piece = v.decode("utf-8")
                elif f == 2 and w == 5:
                    score = struct.unpack("<f", v)[0]
                elif f == 3 and w == 0:
                    ptype = v
            model.pieces.append((piece, score, ptype))
        elif field_number == 2 and wire_type == 2:  # TrainerSpec
            for f, _, v in _iter_fields(value):
                if f == 3 and _ == 0:
                    model.model_type = v
        elif field_number == 3 and wire_type == 2:  # NormalizerSpec
            for f, w, v in _iter_fields(value):
                if f == 1 and w == 2:
                    model.normalizer_name = v.decode("utf-8")
                elif f == 2 and w == 0:
                    model.add_dummy_prefix = bool(v)
    if not model.pieces:
        raise ValueError("SentencePiece model contains no pieces")
    return model


def load_sentencepiece_model(source: Union[str, Path, bytes]) -> SentencePieceModel:
    """Loads a SentencePiece ``.model`` file (path) or raw protobuf bytes."""
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            data = f.read()
    else:
        data = source
    return parse_sentencepiece_proto(data)


def import_sentencepiece(source: Union[str, Path, bytes]) -> CustomTokenizer:
    """
    Imports a SentencePiece **Unigram** ``.model`` into a UniqToken
    :class:`CustomTokenizer`.

    Piece scores (log-probs) and IDs (piece order) are preserved exactly.
    CONTROL/UNKNOWN/USER_DEFINED pieces become UniqToken special tokens; BYTE
    pieces (``<0xXX>``) enable UniqToken's byte fallback. BPE/Word/Char trainer
    models are rejected (UniqToken's engines differ; SentencePiece does not
    store BPE merge tables in the proto).

    .. note::

       **Known parity gap — SPM's add_dummy_prefix.** Most SPM-trained
       models set ``add_dummy_prefix=True``, which prepends a metaspace
       (``\\u2581``) to the input *before* the pre-tokenizer runs. This
       means the first word of every encode gets a free ``\\u2581`` prefix
       and can therefore pick consolidated ``\\u2581word`` pieces during
       Viterbi. UniqToken's pre-tokenizer does not prepend a metaspace, so
       its first word never benefits from those consolidated pieces and
       the segmentation of leading words can differ from SPM's.

       For everything after the first word (chunks that already start
       with ``\\u2581`` from the pre-tokenizer), segmentations are
       byte-for-byte identical — see
       :func:`test_sentencepiece_importer.test_encode_id_parity_for_midtext_words`.
    """
    proto = load_sentencepiece_model(source)
    if proto.model_type != SP_MODEL_UNIGRAM:
        raise NotImplementedError(
            f"only Unigram SentencePiece models are supported, got "
            f"{SP_MODEL_TYPE_NAMES.get(proto.model_type, proto.model_type)!r}"
        )

    vocab: Dict[str, float] = {}
    token_to_id: Dict[str, int] = {}
    id_to_token: Dict[int, str] = {}
    special_tokens: List[str] = []
    has_byte_fallback = False
    unk_token: Optional[str] = None

    for idx, (piece, score, ptype) in enumerate(proto.pieces):
        token_to_id[piece] = idx
        id_to_token[idx] = piece
        if ptype in (SP_CONTROL, SP_UNKNOWN, SP_USER_DEFINED):
            special_tokens.append(piece)
            # specials never compete in the lattice; give a floor score
            vocab.setdefault(piece, -10.0)
            if ptype == SP_UNKNOWN:
                unk_token = piece
        elif ptype == SP_BYTE:
            has_byte_fallback = True
            vocab[piece] = score
        elif ptype == SP_UNUSED:
            continue  # keep the ID hole: piece exists but is never produced
        else:
            vocab[piece] = score

    if proto.add_dummy_prefix:
        _warn_unsupported(
            "normalizer",
            f"add_dummy_prefix=True (name={proto.normalizer_name!r}) — UniqToken never "
            "prepends a metaspace token, so leading-word tokenization may differ",
        )
    normalize_unicode = not proto.normalizer_name.startswith("identity")
    casefold = proto.normalizer_name == "nfkc_cf"
    if normalize_unicode and proto.normalizer_name not in ("nfkc", "nmt_nfkc", "nfkc_cf", ""):
        _warn_unsupported("normalizer", f"normalization rule {proto.normalizer_name!r}")

    if not has_byte_fallback and unk_token is None:
        raise ValueError("SentencePiece import requires byte fallback or an UNKNOWN piece; the model contains neither")

    normalizer = Normalizer(
        space_char="\u2581",
        normalize_unicode=normalize_unicode,
        casefold=casefold,
    )
    pre_tokenizer = RegexPreTokenizer(space_char="\u2581")
    unigram = UnigramModel(
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        special_tokens=special_tokens,
        max_subword_len=max(max(len(t) for t in vocab), 1),
        byte_fallback=has_byte_fallback,
        unk_token=unk_token or "<|unk|>",
    )
    return CustomTokenizer(normalizer=normalizer, pre_tokenizer=pre_tokenizer, model=unigram)


def _warn_unsupported(component: str, detail: str) -> None:
    warnings.warn(
        f"SentencePiece importer: {component} ({detail}); imported tokenizer may tokenize differently than the source.",
        stacklevel=3,
    )
