from __future__ import annotations

import json
import struct
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .byte_codec import ByteFallbackEngine
from .pre_tokenizer import Normalizer
from .tokenizer import CustomTokenizer

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3


class GGUFValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGUFTokenType:
    UNDEFINED = 0
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


def classify_gguf_token_type(token: str, model: Any) -> int:
    """Classifies a vocabulary token string into a GGUF/llama.cpp token type."""
    if ByteFallbackEngine.is_byte_token(token):
        return GGUFTokenType.BYTE
    if token == model.unk_token or token in ("<|unk|>", "<unk>"):
        return GGUFTokenType.UNKNOWN
    special_set = set(getattr(model, "special_tokens", []))
    if (
        token in ("<|bos|>", "<s>", "<|eos|>", "</s>", "<|pad|>", "<pad>", "<|sep|>", "<|cls|>", "<|mask|>")
        or token in special_set
        or (token.startswith("<|") and token.endswith("|>"))
    ):
        if token.startswith("<|user_") or token.startswith("<|custom_"):
            return GGUFTokenType.USER_DEFINED
        return GGUFTokenType.CONTROL
    return GGUFTokenType.NORMAL


class HuggingFaceExporter:
    """
    HuggingFace 'tokenizers' Standard Schema Exporter (v1.0).

    Serializes a CustomTokenizer into the canonical HuggingFace tokenizer.json schema,
    enabling direct integration with transformers.AutoTokenizer.from_pretrained().
    """

    @staticmethod
    def export_to_hf_dict(tokenizer: CustomTokenizer) -> Dict[str, Any]:
        """
        Converts internal model parameters into a HuggingFace v1.0 JSON-compliant dictionary.
        """
        model = tokenizer.model
        normalizer = tokenizer.normalizer

        # 1. Build added_tokens (special tokens)
        added_tokens: List[Dict[str, Any]] = []
        special_set = set(model.special_tokens)
        for token_str, token_id in model.token_to_id.items():
            if token_str in special_set or (token_str.startswith("<|") and token_str.endswith("|>")):
                added_tokens.append(
                    {
                        "id": token_id,
                        "content": token_str,
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    }
                )

        # HF Unigram IDs are array positions. Reject sparse or duplicate IDs
        # instead of emitting a tokenizer whose IDs decode to different tokens.
        sorted_tokens = sorted(model.token_to_id.items(), key=lambda item: item[1])
        ids = [token_id for _, token_id in sorted_tokens]
        if ids != list(range(len(ids))):
            raise ValueError("HuggingFace Unigram export requires contiguous token IDs starting at 0")
        vocab_list = [[tok, model.vocab.get(tok, -10.0)] for tok, _ in sorted_tokens]

        unk_id = model.token_to_id.get(model.unk_token)
        if unk_id is None:
            raise ValueError(
                "HuggingFace Unigram export requires the configured unknown token "
                f"{model.unk_token!r} in the vocabulary"
            )

        # Build a normalizer Sequence mirroring Normalizer.normalize's order:
        # NFKC -> unicode-space map -> punctuation map -> lowercase ->
        # whitespace collapse -> strip.
        normalizers: List[Dict[str, Any]] = []
        if normalizer.normalize_unicode:
            normalizers.append({"type": "NFKC"})
        if normalizer.normalize_unicode_spaces:
            normalizers.append(
                {
                    "type": "Replace",
                    "pattern": {"Regex": "[\\u00A0\\u1680\\u2000-\\u200A\\u202F\\u205F\\u3000]"},
                    "content": " ",
                }
            )
        if normalizer.normalize_punctuation:
            for code, repl in Normalizer.PUNCT_MAP.items():
                normalizers.append({"type": "Replace", "pattern": {"String": chr(code)}, "content": repl})
        if normalizer.lowercase:
            normalizers.append({"type": "Lowercase"})
        if normalizer.collapse_whitespaces:
            normalizers.append({"type": "Replace", "pattern": {"Regex": "[ \\t]+"}, "content": " "})
        if normalizer.strip_whitespace:
            normalizers.append({"type": "Strip"})

        pre = tokenizer.pre_tokenizer
        hf_pre_tokenizers: List[Dict[str, Any]] = []
        if pre.split_digits:
            hf_pre_tokenizers.append({"type": "Digits", "individual_numbers": True})
        hf_pre_tokenizers.append(
            {
                "type": "Metaspace",
                "replacement": normalizer.space_char,
                # UniqToken replaces existing whitespace but does not prepend a
                # metaspace token to text with no leading whitespace.
                "prepend_scheme": "never",
                "split": True,
            }
        )
        if pre.split_punctuation:
            # HF variant enum is capitalized
            hf_pre_tokenizers.append({"type": "Punctuation", "behavior": "Isolated"})
        hf_pre_tokenizer: Dict[str, Any] = (
            hf_pre_tokenizers[0]
            if len(hf_pre_tokenizers) == 1
            else {"type": "Sequence", "pretokenizers": hf_pre_tokenizers}
        )

        unrepresentable = []
        if pre.hex_literals:
            unrepresentable.append("hex_literals")
        if pre.digit_chunk_size is not None:
            unrepresentable.append("digit_chunk_size")
        elif pre.digit_chunking != "greedy" and not (pre.digit_chunking == "single" and pre.split_digits):
            unrepresentable.append("digit_chunking")
        if pre.preset is not None:
            unrepresentable.append("preset")
        if not pre.keep_special_tokens:
            unrepresentable.append("keep_special_tokens=False")
        if normalizer.casefold:
            unrepresentable.append("casefold")
        if unrepresentable:
            warnings.warn(
                "HuggingFace export cannot fully represent this UniqToken pre-tokenizer "
                f"configuration ({', '.join(unrepresentable)}); the exported tokenizer "
                "may tokenize differently from the source model.",
                stacklevel=2,
            )

        hf_schema: Dict[str, Any] = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": {"type": "Sequence", "normalizers": normalizers},
            "pre_tokenizer": hf_pre_tokenizer,
            "post_processor": None,
            "decoder": {
                "type": "Sequence",
                "decoders": [
                    {"type": "ByteFallback"},
                    {
                        "type": "Metaspace",
                        "replacement": normalizer.space_char,
                        "prepend_scheme": "never",
                        "split": True,
                    },
                ],
            },
            "model": {
                "type": "Unigram",
                "unk_id": unk_id,
                "byte_fallback": model.byte_fallback,
                "vocab": vocab_list,
            },
        }

        return hf_schema

    @classmethod
    def save_hf_pretrained(cls, tokenizer: CustomTokenizer, output_dir: Union[str, Path]) -> None:
        """
        Saves tokenizer.json and tokenizer_config.json into output_dir.

        ``tokenizer_config.json`` declares ``tokenizer_class:
        "UniqTokenizerFast"`` and an ``auto_map`` entry pointing at
        ``uniqtoken.hf_adapter.UniqTokenizerFast``, so ``AutoTokenizer`` /
        ``UniqTokenizerFast.from_pretrained`` can reinstate the custom fast
        tokenizer from the exported files on any machine that has ``uniqtoken``
        installed.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        hf_json = cls.export_to_hf_dict(tokenizer)
        with open(out_path / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(hf_json, f, ensure_ascii=False, indent=2)

        specials = tokenizer.model.special_tokens
        named = {
            tokenizer.model.unk_token,
            "<|bos|>" if "<|bos|>" in specials else ("<s>" if "<s>" in specials else None),
            "<|eos|>" if "<|eos|>" in specials else ("</s>" if "</s>" in specials else None),
            "<|pad|>" if "<|pad|>" in specials else ("<pad>" if "<pad>" in specials else None),
        }
        config_json = {
            # The custom fast tokenizer class is reinstated on load; fallbacks
            # (PreTrainedTokenizerFast / generic backends) can still read the
            # repo if uniqtoken is missing.
            "tokenizer_class": "UniqTokenizerFast",
            "model_type": "uniqtoken",
            "auto_map": {
                "AutoTokenizer": "uniqtoken.hf_adapter.UniqTokenizerFast",
            },
            "unk_token": tokenizer.model.unk_token,
            "bos_token": "<|bos|>" if "<|bos|>" in specials else ("<s>" if "<s>" in specials else None),
            "eos_token": "<|eos|>" if "<|eos|>" in specials else ("</s>" if "</s>" in specials else None),
            "pad_token": "<|pad|>" if "<|pad|>" in specials else ("<pad>" if "<pad>" in specials else None),
            "additional_special_tokens": [tok for tok in specials if tok not in named and tok is not None],
            "clean_up_tokenization_spaces": False,
            "model_max_length": 4096,
        }
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            from .chat_template import BUILTIN_TEMPLATES

            config_json["chat_template"] = BUILTIN_TEMPLATES.get(chat_template, chat_template)

        with open(out_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config_json, f, ensure_ascii=False, indent=2)

    @classmethod
    def push_to_hub(
        cls,
        tokenizer: CustomTokenizer,
        repo_id: str,
        token: Optional[str] = None,
        commit_message: str = "Upload UniqToken model",
        private: bool = False,
        **kwargs: Any,
    ) -> str:
        """Uploads the HuggingFace-compatible tokenizer files directly to the Hugging Face Hub.

        Args:
            tokenizer: Trained tokenizer to export and upload.
            repo_id: Hub repository id of the form ``"owner/model"``.
            token: Optional Hub access token used for authentication.
            commit_message: Commit message recorded for the upload commit.
            private: Repository visibility applied when the repo is created. Has no
                effect on an already existing repo; use the Hub UI or
                ``update_repo_visibility`` to change an existing repo.
            **kwargs: Forwarded to ``HfApi.upload_folder`` (e.g. ``allow_patterns``,
                ``revision``).

        Returns:
            The commit URL of the completed synchronous upload, or the PR URL
            string when ``multi_commits=True`` is passed (that mode returns the
            PR URL instead of a ``CommitInfo``).

        Raises:
            ValueError: If ``repo_id`` is not exactly ``"owner/model"`` (two nonempty parts),
                or if ``run_as_future=True`` is passed.
            ImportError: If ``huggingface_hub`` is not installed. Run
                ``pip install "uniqtoken[huggingface]"``.
        """
        if not isinstance(repo_id, str) or repo_id.count("/") != 1 or any(not part for part in repo_id.split("/")):
            raise ValueError(f"repo_id must look like 'owner/model', got {repo_id!r}")
        if kwargs.get("run_as_future"):
            raise ValueError(
                "push_to_hub does not support run_as_future=True: the background upload would read "
                "from a temporary staging directory that is deleted before it completes."
            )
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(
                'huggingface_hub is required for push_to_hub. Run `pip install "uniqtoken[huggingface]"`.'
            ) from exc

        with tempfile.TemporaryDirectory() as tmp_dir:
            cls.save_hf_pretrained(tokenizer, tmp_dir)
            model_card = (
                "---\n"
                "library_name: uniqtoken\n"
                "tags:\n"
                "- tokenizer\n"
                "- unigram\n"
                "---\n"
                f"# {repo_id}\n\n"
                "UniqToken tokenizer exported with `HuggingFaceExporter.save_hf_pretrained`.\n\n"
                f"- vocab_size: {tokenizer.vocab_size}\n"
                f"- byte_fallback: {tokenizer.model.byte_fallback}\n"
                f"- unk_token: {tokenizer.model.unk_token}\n\n"
                "## Usage\n\n"
                "```python\n"
                "from transformers import AutoTokenizer\n\n"
                f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n'
                "```\n"
            )
            Path(tmp_dir, "README.md").write_text(model_card, encoding="utf-8")
            api = HfApi(token=token)
            # Repository visibility is set at creation time: upload_folder accepts no `private` argument.
            api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
            # Synchronous upload returns CommitInfo (a Future only with run_as_future=True,
            # rejected above since it would outlive the temporary staging directory).
            # With multi_commits=True the API instead returns the PR URL string directly.
            commit_info = api.upload_folder(
                repo_id=repo_id, folder_path=tmp_dir, commit_message=commit_message, **kwargs
            )
            if isinstance(commit_info, str):
                return commit_info
            return commit_info.commit_url

    @staticmethod
    def export_to_gguf_dict(tokenizer: CustomTokenizer, model_name: str = "llama") -> Dict[str, Any]:
        """
        Converts UniqToken log-probabilities and vocab tables into the LLaMA.cpp GGUF metadata format.
        Produces:
          - tokenizer.ggml.model: model architecture (default 'llama')
          - tokenizer.ggml.tokens: list of all tokens ordered by contiguous token ID
          - tokenizer.ggml.scores: list of float32 log-probabilities corresponding to each token ID
          - tokenizer.ggml.token_type: list of int32 token type classification codes (1..6)
          - tokenizer.ggml.*_token_id: special token IDs if configured in the model
        """
        model = tokenizer.model
        sorted_tokens = sorted(model.token_to_id.items(), key=lambda item: item[1])
        ids = [token_id for _, token_id in sorted_tokens]
        if ids != list(range(len(ids))):
            raise ValueError("GGUF export requires contiguous token IDs starting at 0")

        tokens: List[str] = []
        scores: List[float] = []
        token_types: List[int] = []

        for tok, _ in sorted_tokens:
            tokens.append(tok)
            score = float(model.vocab.get(tok, -10.0))
            scores.append(score)
            token_types.append(classify_gguf_token_type(tok, model))

        gguf_meta: Dict[str, Any] = {
            "tokenizer.ggml.model": model_name,
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.scores": scores,
            "tokenizer.ggml.token_type": token_types,
        }

        def get_token_id(*candidates: Optional[str]) -> Optional[int]:
            """Returns the first matching token ID among candidate strings, or None."""
            for cand in candidates:
                if cand is not None and cand in model.token_to_id:
                    return model.token_to_id[cand]
            return None

        bos_id = get_token_id("<|bos|>", "<s>")
        if bos_id is not None:
            gguf_meta["tokenizer.ggml.bos_token_id"] = int(bos_id)

        eos_id = get_token_id("<|eos|>", "</s>")
        if eos_id is not None:
            gguf_meta["tokenizer.ggml.eos_token_id"] = int(eos_id)

        unk_id = get_token_id(model.unk_token, "<|unk|>", "<unk>")
        if unk_id is not None:
            gguf_meta["tokenizer.ggml.unknown_token_id"] = int(unk_id)

        pad_id = get_token_id("<|pad|>", "<pad>")
        if pad_id is not None:
            gguf_meta["tokenizer.ggml.padding_token_id"] = int(pad_id)

        return gguf_meta

    @classmethod
    def export_to_gguf(
        cls,
        tokenizer: CustomTokenizer,
        output_path: Optional[Union[str, Path]] = None,
        model_name: str = "llama",
    ) -> bytes:
        """
        Serializes a CustomTokenizer into binary GGUF v3 format containing tokenizer metadata.
        Optionally writes to output_path if provided, and returns the GGUF binary bytes.
        """
        meta = cls.export_to_gguf_dict(tokenizer, model_name=model_name)
        data = bytearray()

        # Header: magic (4B), version (uint32), tensor_count (uint64), metadata_kv_count (uint64)
        data.extend(GGUF_MAGIC)
        data.extend(struct.pack("<IQQ", GGUF_VERSION, 0, len(meta)))

        def pack_str(s: str) -> bytes:
            """Encodes a string as a GGUF length-prefixed UTF-8 byte sequence."""
            s_bytes = s.encode("utf-8")
            return struct.pack("<Q", len(s_bytes)) + s_bytes

        for key, val in meta.items():
            # Key
            data.extend(pack_str(key))

            # Value
            if isinstance(val, bool):
                data.extend(struct.pack("<IB", GGUFValueType.BOOL, 1 if val else 0))
            elif isinstance(val, int):
                if key.endswith("_token_id"):
                    data.extend(struct.pack("<II", GGUFValueType.UINT32, val))
                else:
                    data.extend(struct.pack("<Ii", GGUFValueType.INT32, val))
            elif isinstance(val, float):
                data.extend(struct.pack("<If", GGUFValueType.FLOAT32, val))
            elif isinstance(val, str):
                data.extend(struct.pack("<I", GGUFValueType.STRING))
                data.extend(pack_str(val))
            elif isinstance(val, list):
                data.extend(struct.pack("<I", GGUFValueType.ARRAY))
                if len(val) == 0:
                    data.extend(struct.pack("<IQ", GGUFValueType.STRING, 0))
                elif isinstance(val[0], str):
                    data.extend(struct.pack("<IQ", GGUFValueType.STRING, len(val)))
                    for item in val:
                        data.extend(pack_str(item))
                elif isinstance(val[0], float):
                    data.extend(struct.pack("<IQ", GGUFValueType.FLOAT32, len(val)))
                    data.extend(struct.pack(f"<{len(val)}f", *val))
                elif isinstance(val[0], int):
                    data.extend(struct.pack("<IQ", GGUFValueType.INT32, len(val)))
                    data.extend(struct.pack(f"<{len(val)}i", *val))
                else:
                    raise TypeError(f"Unsupported array element type: {type(val[0])}")
            else:
                raise TypeError(f"Unsupported GGUF metadata value type for key {key!r}: {type(val)}")

        serialized = bytes(data)
        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "wb") as f:
                f.write(serialized)

        return serialized

    @classmethod
    def save_gguf(
        cls,
        tokenizer: CustomTokenizer,
        file_path: Union[str, Path],
        model_name: str = "llama",
    ) -> None:
        """
        Exports the tokenizer and saves it to a .gguf binary file.
        """
        cls.export_to_gguf(tokenizer, output_path=file_path, model_name=model_name)

    @classmethod
    def extract_gguf_metadata(cls, source: Union[str, Path, bytes]) -> Dict[str, Any]:
        """
        Parses a GGUF binary file or bytes and returns the metadata key-value mapping.
        """
        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                data = f.read()
        else:
            data = source

        pos = 0
        if len(data) < 24:
            raise ValueError("Data too short to be a valid GGUF file")

        magic = data[pos : pos + 4]
        pos += 4
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic!r}, expected {GGUF_MAGIC!r}")

        (version,) = struct.unpack("<I", data[pos : pos + 4])
        pos += 4
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        tensor_count, kv_count = struct.unpack("<QQ", data[pos : pos + 16])
        pos += 16

        def read_str(p: int) -> Tuple[str, int]:
            """Reads a GGUF length-prefixed UTF-8 string at buffer offset p."""
            if p + 8 > len(data):
                raise ValueError("Truncated GGUF string length prefix")
            (length,) = struct.unpack("<Q", data[p : p + 8])
            p += 8
            if p + length > len(data):
                raise ValueError("Truncated GGUF string bytes")
            s_bytes = data[p : p + length]
            p += length
            return s_bytes.decode("utf-8", errors="replace"), p

        metadata: Dict[str, Any] = {}
        try:
            for _ in range(kv_count):
                key, pos = read_str(pos)
                if pos + 4 > len(data):
                    raise ValueError("Truncated GGUF metadata value type")
                (val_type,) = struct.unpack("<I", data[pos : pos + 4])
                pos += 4

                if val_type == GGUFValueType.UINT8:
                    (val,) = struct.unpack("<B", data[pos : pos + 1])
                    pos += 1
                elif val_type == GGUFValueType.INT8:
                    (val,) = struct.unpack("<b", data[pos : pos + 1])
                    pos += 1
                elif val_type == GGUFValueType.UINT16:
                    (val,) = struct.unpack("<H", data[pos : pos + 2])
                    pos += 2
                elif val_type == GGUFValueType.INT16:
                    (val,) = struct.unpack("<h", data[pos : pos + 2])
                    pos += 2
                elif val_type == GGUFValueType.UINT32:
                    (val,) = struct.unpack("<I", data[pos : pos + 4])
                    pos += 4
                elif val_type == GGUFValueType.INT32:
                    (val,) = struct.unpack("<i", data[pos : pos + 4])
                    pos += 4
                elif val_type == GGUFValueType.FLOAT32:
                    (val,) = struct.unpack("<f", data[pos : pos + 4])
                    pos += 4
                elif val_type == GGUFValueType.BOOL:
                    (val_b,) = struct.unpack("<B", data[pos : pos + 1])
                    val = bool(val_b)
                    pos += 1
                elif val_type == GGUFValueType.STRING:
                    val, pos = read_str(pos)
                elif val_type == GGUFValueType.ARRAY:
                    if pos + 12 > len(data):
                        raise ValueError("Truncated GGUF array header")
                    item_type, count = struct.unpack("<IQ", data[pos : pos + 12])
                    pos += 12
                    if item_type == GGUFValueType.STRING:
                        arr: List[Any] = []
                        for _ in range(count):
                            s, pos = read_str(pos)
                            arr.append(s)
                    elif item_type == GGUFValueType.FLOAT32:
                        needed = count * 4
                        if pos + needed > len(data):
                            raise ValueError("Truncated GGUF float array payload")
                        arr = list(struct.unpack(f"<{count}f", data[pos : pos + needed]))
                        pos += needed
                    elif item_type == GGUFValueType.INT32:
                        needed = count * 4
                        if pos + needed > len(data):
                            raise ValueError("Truncated GGUF int32 array payload")
                        arr = list(struct.unpack(f"<{count}i", data[pos : pos + needed]))
                        pos += needed
                    elif item_type == GGUFValueType.UINT32:
                        needed = count * 4
                        if pos + needed > len(data):
                            raise ValueError("Truncated GGUF uint32 array payload")
                        arr = list(struct.unpack(f"<{count}I", data[pos : pos + needed]))
                        pos += needed
                    elif item_type == GGUFValueType.INT64:
                        needed = count * 8
                        if pos + needed > len(data):
                            raise ValueError("Truncated GGUF int64 array payload")
                        arr = list(struct.unpack(f"<{count}q", data[pos : pos + needed]))
                        pos += needed
                    elif item_type == GGUFValueType.UINT64:
                        needed = count * 8
                        if pos + needed > len(data):
                            raise ValueError("Truncated GGUF uint64 array payload")
                        arr = list(struct.unpack(f"<{count}Q", data[pos : pos + needed]))
                        pos += needed
                    elif item_type == GGUFValueType.FLOAT64:
                        needed = count * 8
                        if pos + needed > len(data):
                            raise ValueError("Truncated GGUF float64 array payload")
                        arr = list(struct.unpack(f"<{count}d", data[pos : pos + needed]))
                        pos += needed
                    elif item_type == GGUFValueType.BOOL:
                        if pos + count > len(data):
                            raise ValueError("Truncated GGUF bool array payload")
                        arr = [bool(b) for b in data[pos : pos + count]]
                        pos += count
                    else:
                        raise NotImplementedError(f"Unsupported GGUF array element type {item_type}")
                    val = arr
                elif val_type == GGUFValueType.UINT64:
                    (val,) = struct.unpack("<Q", data[pos : pos + 8])
                    pos += 8
                elif val_type == GGUFValueType.INT64:
                    (val,) = struct.unpack("<q", data[pos : pos + 8])
                    pos += 8
                elif val_type == GGUFValueType.FLOAT64:
                    (val,) = struct.unpack("<d", data[pos : pos + 8])
                    pos += 8
                else:
                    raise NotImplementedError(f"Unsupported GGUF value type {val_type}")

                metadata[key] = val
        except (struct.error, IndexError) as exc:
            raise ValueError(f"Truncated or corrupted GGUF metadata payload: {exc}") from exc

        return metadata

    @classmethod
    def extract_gguf_scores(cls, source: Union[str, Path, bytes]) -> Dict[str, float]:
        """
        Extracts a dictionary mapping token strings to float log-probability scores from GGUF binary data.
        """
        meta = cls.extract_gguf_metadata(source)
        tokens_val = meta.get("tokenizer.ggml.tokens")
        scores_val = meta.get("tokenizer.ggml.scores")
        if (
            not isinstance(tokens_val, list)
            or not isinstance(scores_val, list)
            or not all(isinstance(token, str) for token in tokens_val)
            or not all(isinstance(score, (int, float)) and not isinstance(score, bool) for score in scores_val)
        ):
            raise ValueError(
                "GGUF metadata 'tokenizer.ggml.tokens' must be a list of strings and 'tokenizer.ggml.scores' must be a list of numeric floats"
            )
        if len(tokens_val) != len(scores_val):
            raise ValueError(f"Mismatched tokens ({len(tokens_val)}) and scores ({len(scores_val)}) in GGUF metadata")
        if len(set(tokens_val)) != len(tokens_val):
            raise ValueError("Duplicate tokens detected in GGUF vocabulary table")
        return {token: float(score) for token, score in zip(tokens_val, scores_val)}


# Module-level aliases
GGUFExporter = HuggingFaceExporter
extract_gguf_metadata = HuggingFaceExporter.extract_gguf_metadata
extract_gguf_scores = HuggingFaceExporter.extract_gguf_scores
