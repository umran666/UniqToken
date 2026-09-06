"""Compatibility Engine: load *existing* production tokenizers with exact fidelity.

This module is the stable, non-research surface of UniqToken. It houses the
importers for tiktoken ranks files, HuggingFace ``tokenizer.json`` and
SentencePiece ``.model`` files, and enforces the compat contract:

- exact token ID parity with the source model (no re-ranking, no re-indexing),
- exact pre-tokenization regex of the source model,
- identical segmentation and a zero token count delta,
- imported models are returned **frozen**: any attempt to mutate the
  vocabulary or re-train raises :class:`VocabularyMutationError`.

For training *new* vocabularies (Unigram lattice, SuperBPE, script-aware
seeding), use the Research Engine instead: :mod:`uniqtoken.train`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Optional, Union

from ..bpe_model import BPEModel
from ..byte_codec import ByteFallbackEngine
from ..hf_importer import (
    HFByteLevelBPE,
    import_hf_bpe,
    import_hf_tokenizer,
    import_hf_unigram,
)
from ..pre_tokenizer import Normalizer, RegexPreTokenizer
from ..security_shield import SecurityShield
from ..sentencepiece_importer import (
    import_sentencepiece,
    load_sentencepiece_model,
    parse_sentencepiece_proto,
)
from ..tiktoken_adapter import (
    TIKTOKEN_PATTERNS,
    TiktokenEncoding,
    load_tiktoken_ranks,
)
from ..unigram_trainer import UnigramModel

__all__ = [
    "VocabularyMutationError",
    "FrozenCompatModel",
    "TiktokenCompat",
    "HuggingFaceCompat",
    "SentencePieceCompat",
    "from_tiktoken",
    "from_huggingface",
    "from_sentencepiece",
    "TiktokenEncoding",
    "load_tiktoken_ranks",
    "TIKTOKEN_PATTERNS",
    "HFByteLevelBPE",
    "import_hf_tokenizer",
    "import_hf_unigram",
    "import_hf_bpe",
    "import_sentencepiece",
    "load_sentencepiece_model",
    "parse_sentencepiece_proto",
]


class VocabularyMutationError(TypeError):
    """Raised when a compat-loaded model's vocabulary is mutated or re-ranked."""


#: Model/config types whose state carries token IDs or tokenization behavior.
#: Instances of these are surfaced through recursive frozen views.
_FROZEN_TYPES = (
    UnigramModel,
    BPEModel,
    Normalizer,
    RegexPreTokenizer,
    SecurityShield,
    ByteFallbackEngine,
)


def _freeze_value(value: Any) -> Any:
    """Returns an immutable read-only view of ``value`` when it is mutable."""
    if isinstance(value, dict):
        return MappingProxyType(value)
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, set):
        return frozenset(value)
    if isinstance(value, _FROZEN_TYPES):
        return FrozenCompatModel(value)
    return value


class FrozenCompatModel:
    """
    Read-only view over a compat-imported tokenizer.

    Delegates every read/encode/decode attribute to the wrapped model, but:

    - attribute writes and re-training raise
      :class:`VocabularyMutationError`;
    - mutable delegated state is surfaced through immutable views:
      mappings become :class:`types.MappingProxyType`, lists become tuples,
      sets become frozensets, and nested model/config objects
      (:class:`~uniqtoken.unigram_trainer.UnigramModel`,
      :class:`~uniqtoken.bpe_model.BPEModel`,
      :class:`~uniqtoken.pre_tokenizer.Normalizer`,
      :class:`~uniqtoken.pre_tokenizer.RegexPreTokenizer`,
      :class:`~uniqtoken.security_shield.SecurityShield`,
      :class:`~uniqtoken.byte_codec.ByteFallbackEngine`) are wrapped in
      further frozen views — so token IDs and tokenization behavior can
      never change through this surface.

    ponytail: the freeze is recursive only over the known model/config types
    above; unknown mutable objects nested deeper are still delegated as-is.
    Upgrade path if ever needed: whitelist-only attribute exposure per
    model type.
    """

    __slots__ = ("_model",)

    def __init__(self, model: Any):
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name: str) -> Any:
        return _freeze_value(getattr(self._model, name))

    def __setattr__(self, name: str, value: Any) -> None:
        raise VocabularyMutationError(
            "compat models are frozen: attribute assignment would mutate the "
            "imported vocabulary or configuration. Train a new model via "
            "uniqtoken.train instead."
        )

    def train_from_corpus(self, *args: Any, **kwargs: Any) -> Any:
        raise VocabularyMutationError(
            "compat models are frozen: re-training would change token IDs. "
            "Use uniqtoken.train (UnigramTrainer / BPETrainer) to build a new "
            "vocabulary."
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FrozenCompatModel({self._model!r})"


class TiktokenCompat(TiktokenEncoding):
    """
    Frozen tiktoken-compatible encoding: the Compatibility Engine surface of
    :class:`TiktokenEncoding`.

    Token IDs (ranks) are exposed as a read-only mapping and attribute writes
    are rejected, so an imported encoding can never be re-ranked or mutated.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        object.__setattr__(self, "_frozen", False)
        super().__init__(*args, **kwargs)
        self.__dict__["ranks"] = MappingProxyType(dict(self.ranks))
        self.__dict__["special_tokens"] = MappingProxyType(dict(self.special_tokens))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_frozen"):
            raise VocabularyMutationError(
                "compat models are frozen: attribute assignment would mutate "
                "the imported vocabulary. Train a new model via uniqtoken.train "
                "instead."
            )
        object.__setattr__(self, name, value)


# Frozen loader entry points (issue #49 facade names).
def HuggingFaceCompat(source: Union[str, Path, Dict[str, Any]]) -> FrozenCompatModel:
    """Frozen HuggingFace ``tokenizer.json`` import — see :func:`from_huggingface`."""
    return from_huggingface(source)


def SentencePieceCompat(source: Union[str, Path, bytes]) -> FrozenCompatModel:
    """Frozen SentencePiece ``.model`` import — see :func:`from_sentencepiece`."""
    return from_sentencepiece(source)


def from_tiktoken(
    path: Union[str, Path],
    name: str = "custom",
    pattern: str = "cl100k_base",
    special_tokens: Optional[Dict[str, int]] = None,
    explicit_n_vocab: Optional[int] = None,
) -> FrozenCompatModel:
    """
    Compat entry point: loads a tiktoken ``.tiktoken`` ranks file.

    Token IDs are the ranks themselves, preserved exactly; ``pattern`` is a
    TIKTOKEN_PATTERNS preset name ("gpt2", "cl100k_base", "o200k_base") or a
    raw regex string.
    """
    return FrozenCompatModel(
        TiktokenCompat.from_file(
            path,
            name=name,
            pattern=pattern,
            special_tokens=special_tokens,
            explicit_n_vocab=explicit_n_vocab,
        )
    )


def from_huggingface(source: Union[str, Path, Dict[str, Any]]) -> FrozenCompatModel:
    """
    Compat entry point: loads a HuggingFace ``tokenizer.json``.

    Accepts the already-parsed dict or a path to the ``tokenizer.json`` file.
    Vocab scores and token IDs are preserved exactly.
    """
    data = source
    if isinstance(source, (str, Path)):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    return FrozenCompatModel(import_hf_tokenizer(data))


def from_sentencepiece(source: Union[str, Path, bytes]) -> FrozenCompatModel:
    """
    Compat entry point: loads a SentencePiece Unigram ``.model`` file.

    Piece scores and IDs (piece order) are preserved exactly.
    """
    return FrozenCompatModel(import_sentencepiece(source))
