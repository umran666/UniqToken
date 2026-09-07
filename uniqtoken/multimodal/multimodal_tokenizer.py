from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from uniqtoken.multimodal.image_patcher import DynamicImagePatcher, ImagePatch
from uniqtoken.multimodal.visual_codebook import VisualCodebook
from uniqtoken.multimodal.audio_codec import ResidualVectorQuantizer, AudioSegment
from uniqtoken.tokenizer import CustomTokenizer


@dataclass
class MultimodalSequence:
    """
    Unified representation of an interleaved text + vision + audio sequence.
    """

    token_strings: List[str]
    token_ids: List[int]
    image_patches: List[ImagePatch]
    modality_mask: List[int]  # 0: Text, 1: Vision, 2: Audio, 3: Special


class MultimodalTokenizer:
    """
    Unified Discrete & Continuous Multimodal Tokenizer.

    Supports:
    1. Natural Language & Source Code (Unigram + UTF-8 Byte Fallback).
    2. 2D Visual Images (Dynamic Patching + Discrete VQ Codebook Tokenization).
    3. Spatial Grid & Coordinate Anchors (<|image_start|>, <|image_end|>, <|vis_XXXX|>).
    4. Interleaved Multimodal Document Processing.
    """

    MULTIMODAL_SPECIAL_TOKENS = [
        "<|image_start|>",
        "<|image_end|>",
        "<|image_patch|>",
        "<|audio_start|>",
        "<|audio_end|>",
    ]

    _IMAGE_SIZE_RE = re.compile(r"^<\|img_(\d+)x(\d+)\|>$")
    _GRID_RE = re.compile(r"^<\|grid_(\d+)x(\d+)\|>$")

    def __init__(
        self,
        text_tokenizer: CustomTokenizer,
        patch_size: int = 16,
        channels: int = 3,
        num_visual_tokens: int = 512,
        seed: int = 42,
        ema_decay: float = 0.99,
    ):
        if not isinstance(text_tokenizer, CustomTokenizer):
            raise TypeError(f"text_tokenizer must be a CustomTokenizer instance, got {type(text_tokenizer).__name__}")

        self.text_tokenizer = text_tokenizer
        self.patcher = DynamicImagePatcher(patch_size=patch_size, channels=channels)
        self.codebook = VisualCodebook(
            num_embeddings=num_visual_tokens,
            embedding_dim=patch_size * patch_size * channels,
            seed=seed,
            ema_decay=ema_decay,
        )
        self.audio_quantizer = ResidualVectorQuantizer(
            num_quantizers=4,
            codebook_size=256,
            frame_size=320,
            seed=seed,
        )
        self.multimodal_specials = list(self.MULTIMODAL_SPECIAL_TOKENS)
        self.visual_tokens = self.codebook.get_special_tokens()
        self.audio_tokens = self.audio_quantizer.get_special_tokens()
        self._image_element_type = "list"  # Explicit type for image detection
        self._frozen = False

        # Unified ID space: text -> multimodal specials -> visual tokens -> audio tokens
        self._token_to_id: Dict[str, int] = dict(text_tokenizer.model.token_to_id)
        self._next_id = max(self._token_to_id.values(), default=-1) + 1
        self._id_lock = threading.Lock()
        for tok in self.multimodal_specials:
            if tok not in self._token_to_id:
                self._token_to_id[tok] = self._next_id
                self._next_id += 1
        for tok in self.visual_tokens:
            if tok not in self._token_to_id:
                self._token_to_id[tok] = self._next_id
                self._next_id += 1
        for tok in self.audio_tokens:
            if tok not in self._token_to_id:
                self._token_to_id[tok] = self._next_id
                self._next_id += 1

    def freeze(self) -> None:
        """Freezes the vocabulary, preventing dynamic token registration during training or inference."""
        self._frozen = True

    @property
    def vocab_size(self) -> int:
        """Size required for an embedding table spanning the unified ID space."""
        # _next_id is maintained by _assign_id/__init__/load; max()+1 over the
        # dict would be O(n) per access and can lag behind newly assigned IDs.
        return self._next_id

    def _assign_id(self, token: str) -> int:
        """Returns the ID for a token, registering metadata tokens or enforcing frozen vocabulary."""
        with self._id_lock:
            tid = self._token_to_id.get(token)
            if tid is None:
                if self._frozen:
                    raise KeyError(f"Cannot register new token '{token}' on a frozen MultimodalTokenizer vocabulary.")
                tid = self._next_id
                self._token_to_id[token] = tid
                self._next_id += 1
            return tid

    def encode_image(
        self,
        image_pixels: List[List[List[float]]],
    ) -> Tuple[List[str], List[ImagePatch]]:
        """
        Tokenizes a 2D/3D image into discrete visual codebook tokens bracketed by boundary markers.
        """
        if not isinstance(image_pixels, list):
            raise TypeError(f"image_pixels must be a list, got {type(image_pixels).__name__}")
        if not image_pixels:
            return [], []

        patches, (grid_h, grid_w) = self.patcher.extract_patches(image_pixels)
        if not patches:
            return [], []

        # Quantize each patch to its discrete codebook token
        patch_vectors = [p.pixels for p in patches]
        visual_tokens = self.codebook.quantize_patches(patch_vectors)

        img_h = len(image_pixels)
        img_w = len(image_pixels[0]) if img_h else 0

        # Bracket image sequence with boundary anchors and spatial metadata so
        # the decoder can recover the exact aspect ratio and grid layout.
        token_sequence = (
            [
                "<|image_start|>",
                f"<|img_{img_h}x{img_w}|>",
                f"<|grid_{grid_h}x{grid_w}|>",
            ]
            + visual_tokens
            + ["<|image_end|>"]
        )
        return token_sequence, patches

    def encode_interleaved(
        self,
        elements: List[Union[str, "ImageElement", AudioSegment]],
    ) -> MultimodalSequence:
        """
        Encodes a mixed stream of text, images, and audio into a unified multimodal token stream.
        """
        all_tokens: List[str] = []
        all_patches: List[ImagePatch] = []
        modality_mask: List[int] = []

        for element in elements:
            if isinstance(element, str):
                text_toks = self.text_tokenizer.encode(element, allowed_special=set(self.multimodal_specials))
                for t in text_toks:
                    all_tokens.append(t)
                    is_special = t.startswith("<|") and t.endswith("|>")
                    modality_mask.append(3 if is_special else 0)
            elif isinstance(element, ImageElement):
                img_toks, patches = self.encode_image(element.pixels)
                all_patches.extend(patches)
                for t in img_toks:
                    all_tokens.append(t)
                    if (
                        t in {"<|image_start|>", "<|image_end|>"}
                        or self._IMAGE_SIZE_RE.match(t)
                        or self._GRID_RE.match(t)
                    ):
                        modality_mask.append(3)
                    else:
                        modality_mask.append(1)  # Vision modality
            elif isinstance(element, AudioSegment):
                aud_toks, _ = self.audio_quantizer.encode_audio(element.samples)
                for t in aud_toks:
                    all_tokens.append(t)
                    if t in {"<|audio_start|>", "<|audio_end|>"} or t.startswith("<|aud_len_"):
                        modality_mask.append(3)
                    else:
                        modality_mask.append(2)  # Audio modality
            else:
                raise TypeError(
                    f"elements must contain str, ImageElement, or AudioSegment, got {type(element).__name__}"
                )

        token_ids = [self._assign_id(t) for t in all_tokens]

        return MultimodalSequence(
            token_strings=all_tokens,
            token_ids=token_ids,
            image_patches=all_patches,
            modality_mask=modality_mask,
        )

    def decode_text_and_images(
        self,
        token_strings: List[str],
    ) -> Tuple[str, List[List[List[List[float]]]]]:
        """
        Separates text and reconstructs embedded images from the multimodal token stream.
        """
        text_segments: List[str] = []
        reconstructed_images: List[List[List[List[float]]]] = []

        in_image = False
        current_image_tokens: List[str] = []
        pending_size: Optional[Tuple[int, int]] = None
        pending_grid: Optional[Tuple[int, int]] = None

        for tok in token_strings:
            if tok == "<|image_start|>":
                if in_image:
                    raise ValueError("nested image_start marker in multimodal token stream")
                in_image = True
                current_image_tokens.clear()
                pending_size = None
                pending_grid = None
            elif tok == "<|image_end|>":
                if not in_image:
                    raise ValueError("image_end marker without a matching image_start")
                if not current_image_tokens:
                    raise ValueError("image stream contains no visual tokens")
                in_image = False
                # Reconstruct image from visual tokens
                if current_image_tokens:
                    reconstructed_images.append(
                        self._reconstruct_image(
                            current_image_tokens,
                            original_size=pending_size,
                            grid=pending_grid,
                        )
                    )
            elif in_image:
                size_match = self._IMAGE_SIZE_RE.match(tok)
                if size_match:
                    pending_size = (int(size_match.group(1)), int(size_match.group(2)))
                    continue
                grid_match = self._GRID_RE.match(tok)
                if grid_match:
                    pending_grid = (int(grid_match.group(1)), int(grid_match.group(2)))
                    continue
                if not tok.startswith("<|vis_") or not tok.endswith("|>"):
                    raise ValueError(f"invalid token inside image stream: {tok!r}")
                current_image_tokens.append(tok)
            else:
                if tok.startswith("<|vis_") or tok.startswith("<|grid_") or tok.startswith("<|img_"):
                    raise ValueError(f"visual token outside image stream: {tok!r}")
                text_segments.append(tok)

        if in_image:
            raise ValueError("unterminated image stream: missing image_end marker")

        # Filter out visual and audio tokens that might have leaked into text
        filtered_text = [
            t
            for t in text_segments
            if not (t.startswith("<|vis_") and t.endswith("|>"))
            and not (t.startswith("<|aud_") and t.endswith("|>"))
            and t not in {"<|audio_start|>", "<|audio_end|>"}
        ]

        # Decode text using the text tokenizer — a missing token means the stream
        # does not match this tokenizer; surface it instead of corrupting text
        # by silently mapping to ID 0.
        text_model_ids = self.text_tokenizer.model.token_to_id
        unknown = [t for t in filtered_text if t not in text_model_ids]
        if unknown:
            raise KeyError(
                f"token(s) {unknown[:5]!r} not found in the text tokenizer vocabulary; "
                "the token stream does not match this tokenizer"
            )
        decoded_text = self.text_tokenizer.decode([text_model_ids[t] for t in filtered_text])
        return decoded_text, reconstructed_images

    def _reconstruct_image(
        self,
        visual_tokens: List[str],
        original_size: Optional[Tuple[int, int]],
        grid: Optional[Tuple[int, int]],
    ) -> List[List[List[float]]]:
        """Rebuilds an image from visual tokens, honoring the encoded spatial layout."""
        p = self.patcher.patch_size
        img_h, img_w = original_size if original_size else (0, 0)

        if grid:
            grid_h, grid_w = grid
            if grid_h <= 0 or grid_w <= 0:
                raise ValueError("image grid dimensions must be positive")
        elif img_h and img_w:
            grid_h = math.ceil(img_h / p)
            grid_w = math.ceil(img_w / p)
        else:
            grid_h = grid_w = math.ceil(math.sqrt(len(visual_tokens)))

        reconstructed_patches: List[ImagePatch] = []
        if img_h <= 0 or img_w <= 0:
            if original_size is not None:
                raise ValueError("image dimensions must be positive")
        if len(visual_tokens) != grid_h * grid_w:
            raise ValueError("image visual token count does not match its declared grid")
        # Dequantized pixels live in normalized space. Restore the original
        # range via the patcher config: exact when pixel_range is explicit.
        # An auto-detected encode-time scale cannot be recovered from tokens
        # alone, so without pixel_range the canvas stays normalized.
        if self.patcher.pixel_range is not None:
            scale = self.patcher.pixel_range[1] - self.patcher.pixel_range[0]
            offset = self.patcher.pixel_range[0]
        else:
            scale, offset = 1.0, 0.0
        valid_tokens = visual_tokens
        for idx, v_tok in enumerate(valid_tokens):
            pixels = self.codebook.dequantize_token(v_tok)
            row = idx // grid_w
            col = idx % grid_w
            reconstructed_patches.append(
                ImagePatch(
                    patch_id=idx,
                    row=row,
                    col=col,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    norm_bbox=(
                        round(row / grid_h, 4),
                        round(col / grid_w, 4),
                        round((row + 1) / grid_h, 4),
                        round((col + 1) / grid_w, 4),
                    ),
                    pixels=pixels,
                    scale=scale,
                    offset=offset,
                )
            )

        canvas = self.patcher.reconstruct_image(reconstructed_patches, grid_h=grid_h, grid_w=grid_w)

        # Crop back to the original image dimensions (removes zero padding)
        if img_h and img_w:
            canvas = [row[:img_w] for row in canvas[:img_h]]
        return canvas

    def save(self, directory: Union[str, Path]) -> None:
        """
        Saves the multimodal tokenizer to a directory.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save text tokenizer
        self.text_tokenizer.save(dir_path / "text_tokenizer")

        # Save multimodal tokenizer config
        mm_config = {
            "patch_size": self.patcher.patch_size,
            "channels": self.patcher.channels,
            "normalize_pixels": self.patcher.normalize_pixels,
            "pixel_range": self.patcher.pixel_range,
            "num_visual_tokens": self.codebook.num_embeddings,
            "seed": self.codebook.seed,
            "ema_decay": self.codebook.ema_decay,
            "multimodal_specials": self.multimodal_specials,
            "visual_tokens": self.visual_tokens,
            "token_to_id": self._token_to_id,
            "codebook_state": self.codebook.get_codebook_state(),
            "audio_codebook_state": self.audio_quantizer.get_state(),
            "frozen": self._frozen,
        }

        with open(dir_path / "multimodal_config.json", "w", encoding="utf-8") as f:
            json.dump(mm_config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> MultimodalTokenizer:
        """
        Loads a multimodal tokenizer from a directory.
        """
        dir_path = Path(directory)

        # Load text tokenizer
        text_tokenizer = CustomTokenizer.load(dir_path / "text_tokenizer")

        # Load multimodal config
        with open(dir_path / "multimodal_config.json", "r", encoding="utf-8") as f:
            mm_config = json.load(f)

        # Create multimodal tokenizer
        mm_tok = cls(
            text_tokenizer=text_tokenizer,
            patch_size=mm_config["patch_size"],
            channels=mm_config["channels"],
            num_visual_tokens=mm_config["num_visual_tokens"],
            seed=mm_config["seed"],
            ema_decay=mm_config.get("ema_decay", 0.99),
        )

        # Restore patcher settings not covered by the constructor (round-trip
        # for non-default normalization configs; defaults match __init__).
        mm_tok.patcher.normalize_pixels = bool(mm_config.get("normalize_pixels", True))
        mm_tok.patcher.pixel_range = mm_config.get("pixel_range")

        # Restore codebook state
        mm_tok.codebook = VisualCodebook.from_state(mm_config["codebook_state"])
        mm_tok.visual_tokens = mm_tok.codebook.get_special_tokens()
        if "audio_codebook_state" in mm_config:
            mm_tok.audio_quantizer = ResidualVectorQuantizer.from_state(mm_config["audio_codebook_state"])
            mm_tok.audio_tokens = mm_tok.audio_quantizer.get_special_tokens()

        # Restore token mappings
        mm_tok._token_to_id = {str(token): int(token_id) for token, token_id in mm_config["token_to_id"].items()}
        mm_tok._next_id = max(mm_tok._token_to_id.values(), default=-1) + 1
        mm_tok._frozen = bool(mm_config.get("frozen", False))

        return mm_tok


@dataclass(frozen=True)
class ImageElement:
    """
    Explicit container for image data in interleaved sequences.
    This replaces the ambiguous 'isinstance(element, list)' check.
    """

    pixels: List[List[List[float]]]
    metadata: Optional[dict[str, object]] = None
