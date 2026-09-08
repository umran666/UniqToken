from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class AudioSegment:
    """
    Structured container for a 1D audio waveform.
    """

    samples: List[float]
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        if not isinstance(self.samples, list):
            raise TypeError(f"samples must be a list of floats, got {type(self.samples).__name__}")
        if not self.samples:
            raise ValueError("samples cannot be empty")
        if not isinstance(self.sample_rate, int) or isinstance(self.sample_rate, bool) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in self.samples):
            raise ValueError("samples must contain only finite numeric values")


class ResidualVectorQuantizer:
    """
    Multi-Stage Residual Vector Quantizer (RVQ) for 1D Continuous Audio Signals.

    Implements hierarchical multi-codebook quantization (EnCodec / SoundStream / Whisper style).
    Slices 1D audio waveforms into temporal frames and quantizes each frame through N_q
    sequential residual codebooks, yielding high acoustic fidelity at low bitrates.
    """

    def __init__(
        self,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        frame_size: int = 320,  # 20ms at 16kHz
        seed: int = 42,
    ):
        if num_quantizers <= 0:
            raise ValueError("num_quantizers must be positive")
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.frame_size = frame_size
        self.seed = seed

        self._rng = random.Random(seed)
        # N_q independent codebooks, each with codebook_size centroids of dimension frame_size
        self.codebooks: List[List[List[float]]] = [
            [
                [self._rng.gauss(0.0, 1.0 / math.sqrt(frame_size)) for _ in range(frame_size)]
                for _ in range(codebook_size)
            ]
            for _ in range(num_quantizers)
        ]

        # Precompute squared norms for efficient dot-product quantization
        self._codebook_norms: List[List[float]] = [[sum(c * c for c in vec) for vec in cb] for cb in self.codebooks]

    def get_special_tokens(self) -> List[str]:
        """
        Returns all discrete RVQ audio token identifiers across all quantizer stages.
        Format: <|aud_q{q}_{k:04d}|>
        """
        tokens: List[str] = []
        for q in range(self.num_quantizers):
            for k in range(self.codebook_size):
                tokens.append(f"<|aud_q{q}_{k:04d}|>")
        return tokens

    def _quantize_stage(self, residual_vector: List[float], stage: int) -> Tuple[int, str, List[float]]:
        """
        Finds the nearest centroid in the stage-th codebook.

        Computes ``||y||^2 - 2<x, y>``, which omits the constant ``||x||^2`` term
        of ``||x - y||^2`` — identical argmin, but the value is NOT a distance.
        """
        cb = self.codebooks[stage]
        norms = self._codebook_norms[stage]

        best_idx = 0
        min_dist = float("inf")

        for idx, (vec, norm_sq) in enumerate(zip(cb, norms)):
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            dot_prod = sum(r * v for r, v in zip(residual_vector, vec))
            dist = norm_sq - 2.0 * dot_prod
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        token_str = f"<|aud_q{stage}_{best_idx:04d}|>"
        centroid_vec = cb[best_idx]
        return best_idx, token_str, centroid_vec

    def encode_audio(self, samples: List[float]) -> Tuple[List[str], int]:
        """
        Encodes a 1D audio sample array into a sequence of discrete hierarchical RVQ tokens.
        Returns (list_of_tokens, num_frames).
        """
        if not isinstance(samples, list):
            raise TypeError(f"samples must be a list of floats, got {type(samples).__name__}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in samples):
            raise ValueError("samples must contain only finite numeric values")
        if not samples:
            raise ValueError("samples cannot be empty")

        # Frame audio samples
        n_samples = len(samples)
        num_frames = math.ceil(n_samples / self.frame_size)
        tokens: List[str] = ["<|audio_start|>", f"<|aud_len_{n_samples}|>"]

        for f_idx in range(num_frames):
            start = f_idx * self.frame_size
            end = min(start + self.frame_size, n_samples)
            frame = samples[start:end]

            # Zero-pad trailing frame
            if len(frame) < self.frame_size:
                frame = frame + [0.0] * (self.frame_size - len(frame))

            current_residual = list(frame)
            for q in range(self.num_quantizers):
                idx, tok, centroid = self._quantize_stage(current_residual, q)
                tokens.append(tok)
                # Subtract quantized vector from residual
                current_residual = [r - c for r, c in zip(current_residual, centroid)]

        tokens.append("<|audio_end|>")
        return tokens, num_frames

    def decode_audio(self, tokens: List[str]) -> List[float]:
        """
        Reconstructs the 1D audio waveform by summing residual codebook centroids.
        """
        if not isinstance(tokens, list):
            raise TypeError(f"tokens must be a list of strings, got {type(tokens).__name__}")
        if not tokens:
            return []

        reconstructed: List[float] = []
        original_length: Optional[int] = None
        seen_start = False
        seen_end = False

        # Validate the complete stream before decoding. Silently skipping an
        # unknown token would shift frame boundaries or discard corrupted data.
        stage_tokens: List[str] = []
        for t in tokens:
            if not isinstance(t, str):
                raise TypeError(f"audio tokens must be strings, got {type(t).__name__}")
            if t == "<|audio_start|>":
                if seen_start or seen_end:
                    raise ValueError("audio stream contains an invalid or duplicate audio_start marker")
                seen_start = True
                continue
            if t == "<|audio_end|>":
                if not seen_start or seen_end:
                    raise ValueError("audio_end marker without a matching audio_start")
                seen_end = True
                continue
            if not seen_start or seen_end:
                raise ValueError(f"audio token appears outside the audio stream: {t!r}")
            if t.startswith("<|aud_len_") and t.endswith("|>"):
                if original_length is not None:
                    raise ValueError("audio stream contains duplicate length metadata")
                try:
                    original_length = int(t[10:-2])
                except ValueError:
                    raise ValueError(f"malformed audio length token {t!r}") from None
                if original_length < 0:
                    raise ValueError(f"audio length must be non-negative, got {original_length}")
                continue
            if t.startswith("<|aud_q") and t.endswith("|>"):
                stage_tokens.append(t)
                continue
            raise ValueError(f"malformed audio token {t!r}")

        if not seen_start or not seen_end:
            raise ValueError("audio stream must contain matching audio_start and audio_end markers")

        # Process tokens in blocks of num_quantizers; strict validation — a
        # silently dropped or misaligned token would shift every subsequent frame
        # and corrupt the audio with no error.
        if len(stage_tokens) % self.num_quantizers != 0:
            raise ValueError(
                f"audio token stream has {len(stage_tokens)} stage tokens, which is "
                f"not a multiple of num_quantizers ({self.num_quantizers})"
            )
        for b in range(0, len(stage_tokens), self.num_quantizers):
            frame_acc = [0.0] * self.frame_size
            for q in range(self.num_quantizers):
                tok = stage_tokens[b + q]
                # Parse stage and code index: <|aud_q{q}_{idx}|>
                parts = tok[7:-2].split("_")
                if len(parts) != 2:
                    raise ValueError(f"malformed audio token {tok!r}")
                try:
                    stage_num = int(parts[0])
                    code_idx = int(parts[1])
                except ValueError:
                    raise ValueError(f"malformed audio token {tok!r}") from None
                if stage_num != q:
                    raise ValueError(
                        f"audio token {tok!r} declares stage {stage_num}, but stream position {b + q} expects stage {q}"
                    )
                if not 0 <= code_idx < self.codebook_size:
                    raise ValueError(f"audio token {tok!r} has out-of-range code index {code_idx}")
                centroid = self.codebooks[q][code_idx]
                frame_acc = [a + c for a, c in zip(frame_acc, centroid)]
            reconstructed.extend(frame_acc)

        if original_length is not None:
            if original_length > len(reconstructed):
                raise ValueError(f"audio length {original_length} exceeds decoded frame capacity {len(reconstructed)}")
            reconstructed = reconstructed[:original_length]

        return reconstructed

    def get_state(self) -> dict:
        """Return a serializable RVQ state for exact save/load reconstruction."""
        return {
            "num_quantizers": self.num_quantizers,
            "codebook_size": self.codebook_size,
            "frame_size": self.frame_size,
            "seed": self.seed,
            "codebooks": self.codebooks,
        }

    @classmethod
    def from_state(cls, state: dict) -> "ResidualVectorQuantizer":
        """Reconstruct an RVQ instance from a serialized state."""
        quantizer = cls(
            num_quantizers=state["num_quantizers"],
            codebook_size=state["codebook_size"],
            frame_size=state["frame_size"],
            seed=state.get("seed", 42),
        )
        codebooks = state["codebooks"]
        if len(codebooks) != quantizer.num_quantizers:
            raise ValueError("audio codebook state has an invalid quantizer count")
        if any(len(codebook) != quantizer.codebook_size for codebook in codebooks):
            raise ValueError("audio codebook state has an invalid codebook size")
        if any(len(vector) != quantizer.frame_size for codebook in codebooks for vector in codebook):
            raise ValueError("audio codebook state has an invalid vector dimension")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for codebook in codebooks
            for vector in codebook
            for value in vector
        ):
            raise ValueError("audio codebook state contains non-finite values")
        # copy: avoid aliasing the caller's nested lists
        quantizer.codebooks = [[vector[:] for vector in codebook] for codebook in codebooks]
        quantizer._codebook_norms = [
            [sum(value * value for value in vector) for vector in codebook] for codebook in quantizer.codebooks
        ]
        return quantizer
