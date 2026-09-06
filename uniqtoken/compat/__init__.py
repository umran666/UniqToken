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
from typing import Any, Dict, Optional, Union

from ..hf_importer import (
    HFByteLevelBPE,
    import_hf_bpe,
    import_hf_tokenizer,
    import_hf_unigram,
)
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


class FrozenCompatModel:
    """
    Read-only view over a compat-imported tokenizer.

    Delegates every read/encode/decode attribute to the wrapped model, but
    rejects attribute writes and re-training with
    :class:`VocabularyMutationError`. This guarantees the compat contract:
    token IDs, ranks and pre-tokenization behavior can never change after
    import.

    ponytail: this is a facade, not deep immutability — direct mutation of the
    wrapped model (``fm._model.model = ...``) bypasses the guard. Upgrade path
    if ever needed: copy-on-wrap with mappingproxy views over vocab structures.
    """

    __slots__ = ("_model",)

    def __init__(self, model: Any):
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

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


# Facade aliases required by the compat contract (issue #49).
TiktokenCompat = TiktokenEncoding
HuggingFaceCompat = import_hf_tokenizer
SentencePieceCompat = import_sentencepiece


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
        TiktokenEncoding.from_file(
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
