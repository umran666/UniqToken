from __future__ import annotations

import math
import os
import unicodedata
import unittest
from math import log

from hypothesis import HealthCheck, errors, given, settings, strategies as st

from uniqtoken.byte_codec import ByteFallbackEngine
from uniqtoken.multimodal.audio_codec import ResidualVectorQuantizer
from uniqtoken.multimodal.image_patcher import DynamicImagePatcher
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.security_shield import SecurityShield
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramModel

# Profile Architecture Note (Issue #46):
# Volumetric generation (~1,000,000+ raw inputs per scheduled run) is executed
# by the native Rust LibFuzzer target (`fuzz_viterbi` under AddressSanitizer).
# The Python Hypothesis suite focuses on algebraic invariant verification, multi-layer
# dual-offset alignment, and state-machine property testing.
settings.register_profile(
    "dev",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
settings.register_profile(
    "default",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
settings.register_profile(
    "ci",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
settings.register_profile(
    "fuzz",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

active_profile = os.getenv("HYPOTHESIS_PROFILE", "default")
try:
    settings.load_profile(active_profile)
except (KeyError, ValueError, errors.InvalidArgument):
    settings.load_profile("default")

# Reusable custom hypothesis strategies
combining_chars = st.characters(categories=["Mn", "Mc", "Me"])
zwj_chars = st.sampled_from(["\u200c", "\u200d", "\ufe0f", "\ufe0e"])
multilingual_alphabets = st.characters(
    blacklist_categories=["Cs"],  # Exclude lone surrogates from unicode text
    blacklist_characters=["\x00"],
)

adversarial_single_char = st.one_of(
    combining_chars,
    zwj_chars,
    multilingual_alphabets,
    st.sampled_from([" ", "\t", "\n", "\r", "–", "—", "“", "”", "…", "ﬁ", "½", "⚡", "a", "b", "c"]),
)

phrase_fragments = st.sampled_from(["Hello", "World", "مرحبا", "नमस्ते", "👨‍👩‍👧‍👦", "देवनागरी", "2024!"])

adversarial_unicode_text = st.lists(
    st.one_of(st.text(alphabet=adversarial_single_char, min_size=1, max_size=15), phrase_fragments),
    min_size=0,
    max_size=6,
).map(lambda parts: "".join(parts))


def _build_test_tokenizer() -> tuple[CustomTokenizer, CustomTokenizer]:
    """Construct standard and lossless CustomTokenizer instances for property tests."""
    vocab: dict[str, float] = {
        "a": log(0.1),
        "b": log(0.1),
        "c": log(0.1),
        "the": log(0.2),
        "test": log(0.2),
        "token": log(0.15),
        "ization": log(0.15),
        "\u2581": log(0.05),
        "<|unk|>": log(0.05),
    }
    # Populate byte fallback vocabulary
    for b in range(256):
        b_tok = ByteFallbackEngine.byte_to_token(b)
        if b_tok not in vocab:
            vocab[b_tok] = log(0.001)

    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
    id_to_token = {idx: tok for tok, idx in token_to_id.items()}

    model = UnigramModel(
        vocab=vocab,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        special_tokens=["<|unk|>"],
        max_subword_len=8,
        byte_fallback=True,
    )
    normalizer = Normalizer()
    pre_tokenizer = RegexPreTokenizer()
    standard_tokenizer = CustomTokenizer(
        normalizer=normalizer,
        pre_tokenizer=pre_tokenizer,
        model=model,
    )

    lossless_normalizer = Normalizer(
        normalize_unicode=False,
        lowercase=False,
        normalize_unicode_spaces=False,
        normalize_punctuation=False,
    )
    lossless_tokenizer = CustomTokenizer(
        normalizer=lossless_normalizer,
        pre_tokenizer=pre_tokenizer,
        model=model,
    )
    return standard_tokenizer, lossless_tokenizer


STANDARD_TOKENIZER, LOSSLESS_TOKENIZER = _build_test_tokenizer()


class HypothesisTokenizerPropertySuite(unittest.TestCase):
    """
    Continuous Property-Based Fuzz Testing Suite for UniqToken Tokenizer (Issue #46).

    Uses Hypothesis to rigorously fuzz:
    1. Invariant roundtrip across infinite Unicode variations.
    2. Exact dual-offset span bounds and monotonicity.
    3. Adversarial control token injection and delimiter smuggling.
    4. Malformed byte sequence rejection and valid byte reconstruction.
    5. Stochastic subword regularization decoding invariance.
    6. Multimodal spatial and temporal boundary invariants.
    """

    # -------------------------------------------------------------------------
    # PROPERTY 1: Invariant Roundtrip Verification
    # -------------------------------------------------------------------------
    @given(text=adversarial_unicode_text)
    def test_hypothesis_normalized_roundtrip_invariant(self, text: str) -> None:
        """Property: decode(encode(x)) == expected_normalized_and_sanitized(x)."""
        token_ids = STANDARD_TOKENIZER.encode_to_ids(text)
        decoded = STANDARD_TOKENIZER.decode(token_ids)

        normalizer = STANDARD_TOKENIZER.normalizer
        sanitized = STANDARD_TOKENIZER.security.sanitize(text, allowed_special="none")
        expected = normalizer.restore_escaped_metaspace(normalizer.normalize(sanitized)).replace(
            normalizer.space_char, " "
        )

        self.assertEqual(
            decoded,
            expected,
            f"Roundtrip failed for text: {text!r}",
        )

    @given(
        text=st.text(
            alphabet=st.characters(
                min_codepoint=0x20,
                max_codepoint=0x07FF,
                categories=["L", "N", "P", "S", "Z"],
            ),
            min_size=0,
            max_size=80,
        )
    )
    def test_hypothesis_lossless_roundtrip_invariant(self, text: str) -> None:
        """
        Property: With normalization disabled, decode(encode(x)) == x
        for all valid UTF-8 character sequences.
        """
        token_ids = LOSSLESS_TOKENIZER.encode_to_ids(text)
        decoded = LOSSLESS_TOKENIZER.decode(token_ids)
        self.assertEqual(
            decoded,
            text,
            f"Lossless roundtrip failed for: {text!r}",
        )

    # -------------------------------------------------------------------------
    # PROPERTY 2: Dual-Offset Coordinate Bounds & Contiguous Monotonicity
    # -------------------------------------------------------------------------
    @given(text=adversarial_unicode_text)
    def test_hypothesis_dual_offset_bounds_and_monotonicity(self, text: str) -> None:
        """
        Property: For any arbitrary input string:
        1. Lossless tokenizer token spans must tile the entire input from index 0
           to len(text) without gaps or holes: span.start == prev_end or
           (span.start == prev_start and span.end == prev_end) for multibyte tokens.
        2. Standard tokenizer pre-tokenization must preserve dual-offset bounds:
           0 <= norm_start <= norm_end <= len(norm) and
           0 <= raw_start <= raw_end <= len(text) monotonically.
        """
        # 1. Exact contiguous tiling verification under lossless tokenization
        lossless_tokens = LOSSLESS_TOKENIZER.encode_with_offsets(text)
        n_raw = len(text)
        if n_raw > 0:
            self.assertGreater(len(lossless_tokens), 0, f"Non-empty text yielded no tokens: {text!r}")
            self.assertEqual(lossless_tokens[0].raw_span[0], 0, f"First span did not start at 0: {text!r}")
            self.assertEqual(lossless_tokens[-1].raw_span[1], n_raw, f"Last span did not end at len(text): {text!r}")

        prev_s = 0
        prev_e = 0
        for i, tok in enumerate(lossless_tokens):
            s, e = tok.raw_span
            self.assertGreaterEqual(s, 0, f"Span start < 0 for token {tok.text!r} in {text!r}")
            self.assertLessEqual(e, n_raw, f"Span end > len(text) for token {tok.text!r} in {text!r}")
            self.assertLessEqual(s, e, f"Span start > end for token {tok.text!r} in {text!r}")
            if i > 0:
                self.assertTrue(
                    s == prev_e or (s == prev_s and e == prev_e),
                    f"Discontinuous hole or illegal overlap at token {i} ({tok.text!r}): "
                    f"prev=({prev_s}, {prev_e}), curr=({s}, {e}) in {text!r}",
                )
            prev_s, prev_e = s, e

        # 2. Dual-offset coordinates (raw_span & norm_span) in pre-tokenization
        norm, norm_alignment = STANDARD_TOKENIZER.normalizer.normalize_with_alignment(text)
        pre_tokens = STANDARD_TOKENIZER.pre_tokenizer.pre_tokenize_with_offsets(norm, norm_alignment)
        prev_norm_end = 0
        for pt in pre_tokens:
            norm_s, norm_e = pt.norm_span
            raw_s, raw_e = pt.raw_span
            self.assertGreaterEqual(norm_s, 0)
            self.assertLessEqual(norm_e, len(norm))
            self.assertLessEqual(norm_s, norm_e)
            self.assertGreaterEqual(norm_s, prev_norm_end, f"Normalized spans out of order: {norm_s} < {prev_norm_end}")
            prev_norm_end = norm_e

            self.assertGreaterEqual(raw_s, 0)
            self.assertLessEqual(raw_e, n_raw)
            self.assertLessEqual(raw_s, raw_e)

    # -------------------------------------------------------------------------
    # PROPERTY 3: Adversarial Delimiter & Token Smuggling Exhaustion
    # -------------------------------------------------------------------------
    @given(
        prefix=st.text(alphabet=st.characters(blacklist_characters=["<", "|", ">"]), max_size=20),
        token=st.sampled_from(["<|endoftext|>", "<|system|>", "<|user|>", "<|im_start|>", "<|im_end|>"]),
        suffix=st.text(alphabet=st.characters(blacklist_characters=["<", "|", ">"]), max_size=20),
        action=st.sampled_from(["escape", "raise"]),
    )
    def test_hypothesis_adversarial_control_token_neutralization(
        self, prefix: str, token: str, suffix: str, action: str
    ) -> None:
        """
        Property: When allowed_special="none", genuine control tokens are
        either neutralized via escape or rejected via ValueError. Zero leaks.
        """
        raw_text = f"{prefix}{token}{suffix}"
        shield = SecurityShield(
            special_tokens=["<|endoftext|>", "<|system|>", "<|user|>", "<|im_start|>", "<|im_end|>"]
        )

        if action == "raise":
            with self.assertRaises(ValueError):
                shield.sanitize(raw_text, allowed_special="none", disallowed_special_action="raise")
        else:
            sanitized = shield.sanitize(raw_text, allowed_special="none", disallowed_special_action="escape")
            self.assertNotIn(token, sanitized)
            # Invariant: no active control tokens match the special token pattern in sanitized output
            self.assertEqual(len(list(shield.pattern.finditer(sanitized))), 0)

    # -------------------------------------------------------------------------
    # PROPERTY 4: Raw Byte Sequence & ByteFallback Robustness
    # -------------------------------------------------------------------------
    @given(data=st.binary(min_size=0, max_size=128))
    def test_hypothesis_byte_fallback_engine_robustness(self, data: bytes) -> None:
        """
        Property: ByteFallbackEngine must either decode valid UTF-8 losslessly
        or reject invalid UTF-8 with a clean UnicodeDecodeError.
        Zero crashes or unhandled exceptions.
        """
        tokens = [f"<0x{b:02X}>" for b in data]

        # Determine UTF-8 validity independently of the system under test
        is_valid_utf8 = True
        try:
            expected_str = data.decode("utf-8")
        except UnicodeDecodeError:
            is_valid_utf8 = False

        if is_valid_utf8:
            decoded_str = ByteFallbackEngine.decode_tokens(tokens)
            self.assertEqual(decoded_str, expected_str)
            self.assertEqual(decoded_str.encode("utf-8"), data)
        else:
            with self.assertRaises(UnicodeDecodeError):
                ByteFallbackEngine.decode_tokens(tokens)

    # -------------------------------------------------------------------------
    # PROPERTY 5: Stochastic Subword Regularization Decoding Invariance
    # -------------------------------------------------------------------------
    @given(
        text=st.text(
            alphabet=st.characters(
                min_codepoint=0x20,
                max_codepoint=0x07FF,
                categories=["L", "N", "P", "Z"],
            ),
            min_size=1,
            max_size=40,
        ),
        alpha=st.floats(min_value=0.1, max_value=1.0),
    )
    def test_hypothesis_subword_regularization_decoding_invariance(self, text: str, alpha: float) -> None:
        """
        Property: Regardless of stochastic sampling in subword regularization,
        decoding the sampled subwords must produce the exact identical text as
        decoding the deterministic Viterbi segmentation.
        """
        det_ids = STANDARD_TOKENIZER.encode_to_ids(text)
        det_decoded = STANDARD_TOKENIZER.decode(det_ids)

        sampled_tokens = STANDARD_TOKENIZER.sample(text, alpha=alpha)
        sampled_ids = [STANDARD_TOKENIZER.model.token_to_id[t] for t in sampled_tokens]
        sampled_decoded = STANDARD_TOKENIZER.decode(sampled_ids)

        self.assertEqual(
            sampled_decoded,
            det_decoded,
            f"Stochastic regularization decoded string diverged for: {text!r}",
        )

    # -------------------------------------------------------------------------
    # PROPERTY 6: Multimodal Spatial & Temporal Boundary Invariants
    # -------------------------------------------------------------------------
    @given(
        h=st.integers(min_value=1, max_value=64),
        w=st.integers(min_value=1, max_value=64),
        patch_size=st.sampled_from([8, 16]),
    )
    def test_hypothesis_image_patcher_geometry(self, h: int, w: int, patch_size: int) -> None:
        """Property: DynamicImagePatcher preserves exact grid tiling and normalized bbox constraints."""
        patcher = DynamicImagePatcher(patch_size=patch_size, channels=3)
        fake_image = [[[0.5, 0.5, 0.5] for _ in range(w)] for _ in range(h)]

        patches, (grid_h, grid_w) = patcher.extract_patches(fake_image)

        expected_grid_h = math.ceil(h / patch_size)
        expected_grid_w = math.ceil(w / patch_size)

        self.assertEqual(grid_h, expected_grid_h)
        self.assertEqual(grid_w, expected_grid_w)
        self.assertEqual(len(patches), expected_grid_h * expected_grid_w)

        for p in patches:
            y1, x1, y2, x2 = p.norm_bbox
            self.assertGreaterEqual(y1, 0.0)
            self.assertGreaterEqual(x1, 0.0)
            self.assertLessEqual(y2, 1.0)
            self.assertLessEqual(x2, 1.0)
            self.assertLessEqual(y1, y2)
            self.assertLessEqual(x1, x2)

    @given(n_samples=st.integers(min_value=1, max_value=1500))
    def test_hypothesis_audio_rvq_framing(self, n_samples: int) -> None:
        """Property: ResidualVectorQuantizer frames into ceil(T/F) and preserves sample length."""
        frame_size = 320
        rvq = ResidualVectorQuantizer(num_quantizers=4, codebook_size=64, frame_size=frame_size)
        fake_audio = [0.1 * (i % 10) for i in range(n_samples)]

        tokens, num_frames = rvq.encode_audio(fake_audio)
        expected_frames = math.ceil(n_samples / frame_size)
        self.assertEqual(num_frames, expected_frames)

        reconstructed = rvq.decode_audio(tokens)
        self.assertEqual(len(reconstructed), n_samples)


if __name__ == "__main__":
    unittest.main()
