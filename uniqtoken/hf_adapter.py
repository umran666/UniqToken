"""Native HuggingFace ``PreTrainedTokenizerFast`` adapter for UniqToken.

This module exposes :class:`UniqTokenizerFast`, a drop-in
``transformers.PreTrainedTokenizerFast`` subclass that is driven by the
canonical ``tokenizer.json`` produced by
:class:`uniqtoken.hf_exporter.HuggingFaceExporter`. It gives UniqToken models
the full standard fast-tokenizer surface - ``save_pretrained()`` /
``from_pretrained()``, padding & truncation strategies, ``return_tensors``
(``"np"`` / ``"pt"`` / ``"tf"``), batched encoding, offset mappings and
``Trainer`` compatibility - without any custom glue in user code.

Wiring back to the HuggingFace ecosystem
----------------------------------------
Importing this module registers ``UniqTokenizerFast`` with
``transformers.AutoTokenizer`` (via ``REGISTERED_TOKENIZER_CLASSES`` on
transformers >= 5). Combined with the ``tokenizer_class: "UniqTokenizerFast"``
and ``auto_map`` entries written into ``tokenizer_config.json`` by
:meth:`UniqTokenizerFast.save_pretrained` / ``HuggingFaceExporter``, a caller
that has ``uniqtoken`` installed can re-load a UniqToken repo as any other Hub
tokenizer::

    save_dir = tok.save_pretrained("uniqtok_export/")          # writes tokenizer.json + configs
    reloaded = UniqTokenizerFast.from_pretrained("uniqtok_export/")
    auto = AutoTokenizer.from_pretrained("uniqtok_export/")    # -> UniqTokenizerFast
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

from .hf_exporter import HuggingFaceExporter
from .tokenizer import CustomTokenizer

try:
    # ``transformers`` is an optional extra (``uniqtoken[huggingface]``); the
    # adapter is inert when it is missing so the rest of the package keeps
    # importing in a bare environment.
    from transformers import PreTrainedTokenizerFast

    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover - exercised in environments without transformers
    PreTrainedTokenizerFast = object  # type: ignore[assignment, misc]
    HAS_TRANSFORMERS = False

__all__ = ["HAS_TRANSFORMERS", "UniqTokenizerFast", "register_tokenizer"]

#: Bos/eos/pad candidates probed in model preference order (mirror the model's
#: own conventions in ``chat_template.py`` and the GGUF exporter).
_BOS_CANDIDATES = ("<|bos|>", "<s>", "<|begin_of_text|>")
_EOS_CANDIDATES = ("<|eos|>", "</s>", "<|end_of_text|>")
_PAD_CANDIDATES = ("<|pad|>", "<pad>")

#: Default ``model_max_length`` for tokenizer-only exports; callers can pass an
#: explicit ``model_max_length`` to ``from_custom_tokenizer``.
DEFAULT_MODEL_MAX_LENGTH = 4096


def _first_special(special_tokens: List[str], candidates: Any) -> Optional[str]:
    """Return the first configured special token matching ``candidates``, else None."""
    for candidate in candidates:
        if candidate in special_tokens:
            return candidate
    return None


if HAS_TRANSFORMERS:

    class UniqTokenizerFast(PreTrainedTokenizerFast):
        """HuggingFace fast-tokenizer wrapper around a UniqToken model.

        Instances are ordinary ``PreTrainedTokenizerFast`` objects whose
        in-memory backend is the ``tokenizers.Tokenizer`` reconstructed from
        the canonical UniqToken ``tokenizer.json`` export.  All encoding,
        padding, truncation, tensor conversion and serialization behavior is
        therefore the standard HuggingFace fast-tokenizer behavior.
        """

        #: Written into ``tokenizer_config.json`` by ``save_pretrained`` so the
        #: repo advertises the exact class that must be instantiated.
        _auto_map: Any = {"AutoTokenizer": "uniqtoken.hf_adapter.UniqTokenizerFast"}

        @classmethod
        def from_custom_tokenizer(
            cls, tokenizer: CustomTokenizer, model_max_length: Optional[int] = None, **kwargs: Any
        ) -> "UniqTokenizerFast":
            """Build a fast-tokenizer adapter from a trained :class:`CustomTokenizer`.

            The ``tokenizers`` backend is reconstructed from
            ``HuggingFaceExporter.export_to_hf_dict``; special tokens, chat
            template and serialization metadata are carried over so the
            returned tokenizer behaves like any other HF fast tokenizer.

            Args:
                tokenizer: Trained UniqToken tokenizer to wrap.
                model_max_length: Sequence length cap stored in the config and
                    used as the default truncation ceiling. Defaults to
                    :data:`DEFAULT_MODEL_MAX_LENGTH`.
                **kwargs: Extra ``PreTrainedTokenizerFast`` init kwargs,
                    overriding any inferred default.

            Returns:
                A freshly constructed :class:`UniqTokenizerFast`.
            """
            from tokenizers import Tokenizer

            if not isinstance(tokenizer, CustomTokenizer):
                raise TypeError(f"tokenizer must be a uniqtoken.CustomTokenizer, got {type(tokenizer).__name__}")
            hf_dict = HuggingFaceExporter.export_to_hf_dict(tokenizer)
            backend = Tokenizer.from_str(json.dumps(hf_dict, ensure_ascii=False, sort_keys=True))
            specials = list(tokenizer.model.special_tokens)
            defaults: Dict[str, Any] = {"model_max_length": model_max_length or DEFAULT_MODEL_MAX_LENGTH}
            unk = tokenizer.model.unk_token
            if unk:
                defaults["unk_token"] = unk
            bos = _first_special(specials, _BOS_CANDIDATES)
            eos = _first_special(specials, _EOS_CANDIDATES)
            pad = _first_special(specials, _PAD_CANDIDATES)
            if bos:
                defaults["bos_token"] = bos
                defaults["add_bos_token"] = True
            if eos:
                defaults["eos_token"] = eos
                defaults["add_eos_token"] = True
            if pad:
                defaults["pad_token"] = pad
            # HF convention: ``add_special_tokens=True`` prepends/appends bos/eos.
            # Insertion is driven by the backend post-processor, not the
            # ``add_*_token`` flags (some transformers versions auto-build one,
            # some do not) - so build it explicitly here. ``save_pretrained``
            # serializes it into tokenizer.json, so the behavior survives reload.
            if bos or eos:
                from tokenizers.processors import TemplateProcessing

                special = []
                if bos:
                    special.append((bos, backend.token_to_id(bos)))
                if eos:
                    special.append((eos, backend.token_to_id(eos)))
                prefix = f"{bos} " if bos else ""
                suffix = f" {eos}" if eos else ""
                backend.post_processor = TemplateProcessing(
                    single=f"{prefix}$A{suffix}",
                    pair=f"{prefix}$A:0 $B:1{suffix}",
                    special_tokens=special,
                )
            named = {tok for tok in (unk, bos, eos, pad) if tok}
            extra = [tok for tok in specials if tok not in named]
            if extra:
                defaults["additional_special_tokens"] = extra
            defaults["clean_up_tokenization_spaces"] = False
            # Persist the chat template so ``apply_chat_template`` works on the
            # HF side after ``save_pretrained`` / ``from_pretrained``. Built-in
            # template *names* are expanded to their Jinja source.
            chat_template = getattr(tokenizer, "chat_template", None)
            if chat_template:
                from .chat_template import BUILTIN_TEMPLATES

                defaults["chat_template"] = BUILTIN_TEMPLATES.get(chat_template, chat_template)

            defaults.update(kwargs)
            return cls(tokenizer_object=backend, **defaults)

        @classmethod
        def from_pretrained(  # type: ignore[override]
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            *init_inputs: Any,
            **kwargs: Any,
        ) -> "UniqTokenizerFast":
            """Load a saved tokenizer exactly like ``PreTrainedTokenizerFast``.

            Also accepts directories written by
            ``HuggingFaceExporter.save_hf_pretrained`` /
            ``CustomTokenizer.export_to_huggingface`` (a subset of the full
            config keys), which the base implementation handles natively.
            """
            return super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)

else:

    class UniqTokenizerFast:  # type: ignore[no-redef]
        """Placeholder raised when ``transformers`` is not installed."""

        _auto_map: Any = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "uniqtoken.hf_adapter.UniqTokenizerFast requires 'transformers'. "
                'Run `pip install "uniqtoken[huggingface]"`.'
            )


def register_tokenizer() -> bool:
    """Register ``UniqTokenizerFast`` with ``transformers.AutoTokenizer``.

    On transformers >= 5 the class is added to
    ``tokenization_auto.REGISTERED_TOKENIZER_CLASSES`` so that
    ``AutoTokenizer.from_pretrained`` resolves a repo whose
    ``tokenizer_config.json`` declares ``tokenizer_class: "UniqTokenizerFast"``
    to this exact class (no ``trust_remote_code`` required).  Safe to call more
    than once.  Returns ``True`` when registration happened.

    Older transformers releases that lack the registration hook have no
    AutoTokenizer plugin point; call :meth:`UniqTokenizerFast.from_pretrained`
    directly on those versions.
    """
    if not HAS_TRANSFORMERS:
        return False
    try:
        from transformers.models.auto.tokenization_auto import REGISTERED_TOKENIZER_CLASSES  # type: ignore[attr-defined]
    except (ImportError, AttributeError):  # pragma: no cover - pre-v5 transformers
        return False
    REGISTERED_TOKENIZER_CLASSES[UniqTokenizerFast.__name__] = UniqTokenizerFast  # type: ignore[attr-defined]
    return True


# Importing the module wires the class back into the ecosystem once.
if HAS_TRANSFORMERS:  # pragma: no cover - depends on the environment
    register_tokenizer()
