from __future__ import annotations

import json
import math
import struct
import unittest
from math import log
from pathlib import Path
from tempfile import TemporaryDirectory

from uniqtoken.byte_codec import ByteFallbackEngine
from uniqtoken.batch_collator import BatchCollator
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer, TokenizationReport
from uniqtoken.unigram_trainer import UnigramModel, UnigramTrainer
from uniqtoken.unigram_lattice import UnigramLattice
from uniqtoken.vocab_adapter import VocabularyAdapter
from uniqtoken.multimodal.multimodal_tokenizer import (
    ImageElement,
    MultimodalTokenizer,
    TextElement,
)
from uniqtoken.multimodal.visual_codebook import VisualCodebook
from uniqtoken.multimodal.audio_codec import ResidualVectorQuantizer, AudioSegment
from uniqtoken.trie import PrefixTrie
from uniqtoken.bpe_trainer import BPETrainer
from uniqtoken.hf_exporter import (
    GGUFExporter,
    GGUFTokenType,
    HuggingFaceExporter,
    extract_gguf_metadata,
    extract_gguf_scores,
)
from uniqtoken.indentation_compressor import IndentationCompressor
from uniqtoken.security_shield import SecurityShield
from uniqtoken.seed_builder import SeedVocabularyBuilder
from uniqtoken.streaming_decoder import StreamingDecoder
from uniqtoken.cem_merger import CrossEntropyMerging


class NormalizerTests(unittest.TestCase):
    def test_nfkc_composes_across_codepoints_and_preserves_raw_span(self):
        raw = "A\u030a"
        normalized, alignment = Normalizer().normalize_with_alignment(raw)

        self.assertEqual(normalized, "\u00c5")
        tokens = RegexPreTokenizer().pre_tokenize_with_offsets(normalized, alignment)
        self.assertEqual(tokens[0].raw_span, (0, 2))

    def test_whitespace_options_are_applied(self):
        normalized, alignment = Normalizer(
            collapse_whitespaces=True,
            strip_whitespace=True,
        ).normalize_with_alignment("  a\t  b  ")

        self.assertEqual(normalized, "a\u2581b")
        self.assertEqual(alignment[1], (3, 6))

    def test_rejects_misaligned_offset_map(self):
        with self.assertRaises(ValueError):
            RegexPreTokenizer().pre_tokenize_with_offsets("abc", [(0, 1)])

    def test_escapes_literal_metaspace_and_escape_prefix(self):
        raw = "x\u2581y\ue000z"
        normalizer = Normalizer()
        normalized, alignment = normalizer.normalize_with_alignment(raw)

        self.assertEqual(normalized, "x\ue000\ue001y\ue000\ue000z")
        self.assertEqual(normalizer.restore_escaped_metaspace(normalized), raw)
        self.assertEqual(alignment[1:3], [(1, 2), (1, 2)])


class ByteFallbackTests(unittest.TestCase):
    def test_byte_decoding_preserves_literal_metaspace(self):
        tokens = ByteFallbackEngine.char_to_byte_tokens("\u2581")
        self.assertEqual(ByteFallbackEngine.decode_tokens(tokens), "\u2581")

    def test_invalid_byte_sequence_is_rejected(self):
        with self.assertRaises(UnicodeDecodeError):
            ByteFallbackEngine.decode_tokens(["<0xFF>"])

    def test_subwords_still_decode_metaspace(self):
        self.assertEqual(ByteFallbackEngine.decode_tokens(["hello\u2581world"]), "hello world")


class CustomTokenizerTests(unittest.TestCase):
    def setUp(self):
        vocab = {"tok": log(0.5), "en": log(0.3), "ize": log(0.2)}
        token_to_id = {token: index for index, token in enumerate(vocab)}
        self.model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=[],
            max_subword_len=3,
            byte_fallback=False,
        )
        self.tokenizer = CustomTokenizer(
            normalizer=Normalizer(normalize_unicode=False),
            pre_tokenizer=RegexPreTokenizer(),
            model=self.model,
        )

    def test_subword_offsets_are_exact(self):
        tokens = self.tokenizer.encode_with_offsets("tokenize")
        self.assertEqual(
            [(token.text, token.raw_span) for token in tokens],
            [("tok", (0, 3)), ("en", (3, 5)), ("ize", (5, 8))],
        )

    def test_save_load_preserves_lattice_settings(self):
        custom_tokenizer = CustomTokenizer(
            normalizer=Normalizer(
                lowercase=True,
                normalize_unicode=False,
                collapse_whitespaces=True,
            ),
            pre_tokenizer=RegexPreTokenizer(
                split_digits=True,
                split_punctuation=False,
                keep_special_tokens=False,
                special_token_pattern=r"\[\[[^\]]+\]\]",
            ),
            model=self.model,
        )
        with TemporaryDirectory() as directory:
            custom_tokenizer.save(directory)
            loaded = CustomTokenizer.load(directory)

        self.assertEqual(loaded.model.max_subword_len, 3)
        self.assertFalse(loaded.model.byte_fallback)
        self.assertTrue(loaded.normalizer.lowercase)
        self.assertFalse(loaded.normalizer.normalize_unicode)
        self.assertTrue(loaded.normalizer.collapse_whitespaces)
        self.assertTrue(loaded.pre_tokenizer.split_digits)
        self.assertFalse(loaded.pre_tokenizer.split_punctuation)
        self.assertFalse(loaded.pre_tokenizer.keep_special_tokens)
        self.assertEqual(loaded.pre_tokenizer.special_token_pattern, r"\[\[[^\]]+\]\]")

    def test_vocabulary_adapter_preserves_model_settings_and_ids(self):
        updated = VocabularyAdapter.expand_vocabulary(
            self.tokenizer,
            ["tokenized tokenized"],
            num_new_tokens=1,
            min_frequency=1,
            max_ngram_length=8,
            verbose=False,
        )

        added_tokens = set(updated.model.vocab) - set(self.model.vocab)
        self.assertTrue(added_tokens)
        self.assertEqual(updated.model.max_subword_len, max(3, max(map(len, added_tokens))))
        self.assertFalse(updated.model.byte_fallback)
        self.assertEqual(updated.model.token_to_id["tok"], self.model.token_to_id["tok"])
        self.assertEqual(updated.model.token_to_id["en"], self.model.token_to_id["en"])

    def test_vocabulary_adapter_zero_additions_is_a_noop(self):
        self.assertIs(
            VocabularyAdapter.expand_vocabulary(self.tokenizer, ["new domain"], num_new_tokens=0),
            self.tokenizer,
        )

    def test_vocabulary_adapter_compaction_produces_contiguous_ids(self):
        compacted, remap = VocabularyAdapter.compact_vocabulary(self.tokenizer)
        self.assertEqual(len(compacted.model.token_to_id), len(self.model.token_to_id))
        self.assertEqual(sorted(compacted.model.id_to_token.keys()), list(range(len(self.model.token_to_id))))
        self.assertEqual(len(remap), len(self.model.token_to_id))

    def test_trie_clear_seg_cache(self):
        if hasattr(self.model, "_rust_trie") and self.model._rust_trie is not None:
            self.model._rust_trie.clear_seg_cache()
            self.assertEqual(self.model._rust_trie.seg_cache_len(), 0)


class LatticeTests(unittest.TestCase):
    def test_rejects_invalid_sampling_temperature(self):
        lattice = UnigramLattice("ab", {"a": log(0.5), "b": log(0.5)}, byte_fallback=False)
        with self.assertRaises(ValueError):
            lattice.sample(alpha=0)
        with self.assertRaises(ValueError):
            lattice.sample(alpha=float("nan"))

    def test_forward_backward_rejects_disconnected_lattice(self):
        lattice = UnigramLattice("z", {}, byte_fallback=False)
        with self.assertRaises(ValueError):
            lattice.forward_backward()

    def test_rejects_invalid_lattice_length(self):
        with self.assertRaises(ValueError):
            UnigramLattice("a", {"a": log(1.0)}, max_subword_len=0)


class TrainerValidationTests(unittest.TestCase):
    def test_rejects_invalid_training_configuration(self):
        with self.assertRaises(ValueError):
            UnigramTrainer(prune_rate=0)
        with self.assertRaises(ValueError):
            UnigramTrainer(em_sub_iterations=0)
        with self.assertRaises(ValueError):
            UnigramTrainer(max_ngram_length=0)


class BatchCollatorTests(unittest.TestCase):
    def setUp(self):
        vocab = {
            "a": log(0.2),
            "<|pad|>": log(0.2),
            "<|bos|>": log(0.2),
            "<|eos|>": log(0.2),
            "<|unk|>": log(0.2),
        }
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
            byte_fallback=False,
        )
        self.collator = BatchCollator(CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model))

    def test_padding_keeps_tokens_aligned_with_ids(self):
        batch = self.collator.batch_encode(["a", "aa"], max_length=5, truncation=True)
        self.assertEqual(batch.tokens[0], ["<|bos|>", "a", "<|eos|>", "<|pad|>", "<|pad|>"])
        self.assertEqual([len(row) for row in batch.input_ids], [5, 5])
        self.assertEqual([len(row) for row in batch.tokens], [5, 5])
        self.assertEqual(batch.attention_mask[0], [1, 1, 1, 0, 0])

    def test_rejects_overlong_sequence_without_truncation(self):
        with self.assertRaises(ValueError):
            self.collator.batch_encode(["aa"], max_length=2, truncation=False)


class MultimodalTests(unittest.TestCase):
    def setUp(self):
        vocab = {"test": log(0.5), "<|unk|>": log(0.5)}
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=["<|unk|>"],
            byte_fallback=False,
        )
        self.tokenizer = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
        self.mm_tok = MultimodalTokenizer(self.tokenizer, patch_size=16, channels=3, num_visual_tokens=64)

    def test_image_patching_and_aspect_ratio(self):
        img = [[[1.0, 2.0, 3.0] for _ in range(32)] for _ in range(16)]
        tokens, patches = self.mm_tok.encode_image(img)
        self.assertEqual(tokens[0], "<|image_start|>")
        self.assertEqual(tokens[-1], "<|image_end|>")
        self.assertEqual(len(patches), 2)
        self.assertEqual(patches[0].norm_bbox, (0.0, 0.0, 1.0, 0.5))

    def test_interleaved_modality_mask(self):
        img = [[[0.5, 0.5, 0.5] for _ in range(16)] for _ in range(16)]
        img_elem = ImageElement(pixels=img)
        seq = self.mm_tok.encode_interleaved(["test", img_elem, "test"])
        self.assertIn(0, seq.modality_mask)  # Text
        self.assertIn(1, seq.modality_mask)  # Vision
        self.assertIn(3, seq.modality_mask)  # Special

    def test_save_load_preserves_multimodal_state(self):
        img = [[[0.5, 0.5, 0.5] for _ in range(16)] for _ in range(16)]
        img_elem = ImageElement(pixels=img)
        seq = self.mm_tok.encode_interleaved(["test", img_elem, "test"])
        self.mm_tok.audio_quantizer.codebooks[0][0][0] = 123.0

        with TemporaryDirectory() as directory:
            self.mm_tok.save(directory)
            loaded = MultimodalTokenizer.load(directory)

            self.assertEqual(loaded.vocab_size, self.mm_tok.vocab_size)
            self.assertEqual(loaded.codebook.num_embeddings, self.mm_tok.codebook.num_embeddings)
            self.assertEqual(
                loaded.audio_quantizer.codebooks[0][0][0],
                123.0,
            )
            seq2 = loaded.encode_interleaved(["test", img_elem, "test"])
            self.assertEqual(len(seq2.token_strings), len(seq.token_strings))

    def test_freeze_state_survives_save_load(self):
        self.mm_tok.freeze()
        with TemporaryDirectory() as directory:
            self.mm_tok.save(directory)
            loaded = MultimodalTokenizer.load(directory)
        with self.assertRaises(KeyError):
            loaded._assign_id("<|new_metadata|>")

    def test_nonzero_pixel_range_uses_normalized_zero_padding(self):
        from uniqtoken.multimodal.image_patcher import DynamicImagePatcher

        patcher = DynamicImagePatcher(patch_size=2, channels=1, pixel_range=(10.0, 20.0))
        patches, _ = patcher.extract_patches([[[15.0]]])
        self.assertEqual(patches[0].pixels, [0.5, 0.0, 0.0, 0.0])

    def test_rejects_zero_sized_image_grid_metadata(self):
        with self.assertRaises(ValueError):
            self.mm_tok.decode_text_and_images(["<|image_start|>", "<|grid_0x0|>", "<|vis_0000|>", "<|image_end|>"])

    def test_rejects_invalid_element_type(self):
        from typing import Any, cast

        with self.assertRaises(TypeError):
            self.mm_tok.encode_interleaved(cast(Any, [123]))

    def test_codebook_training_updates_embeddings(self):
        img = [[[0.5, 0.5, 0.5] for _ in range(16)] for _ in range(16)]
        initial_codebook = [row[:] for row in self.mm_tok.codebook.codebook]

        tokens, patches = self.mm_tok.encode_image(img)
        patch_vectors = [p.pixels for p in patches]
        indices = [self.mm_tok.codebook.quantize_patch(vec)[0] for vec in patch_vectors]

        for _ in range(101):
            self.mm_tok.codebook.update_ema(patch_vectors, indices)

        updated = False
        for init_row, curr_row in zip(initial_codebook, self.mm_tok.codebook.codebook):
            for iv, cv in zip(init_row, curr_row):
                if abs(iv - cv) > 1e-10:
                    updated = True
                    break
            if updated:
                break

        self.assertTrue(updated, "Codebook should have been updated by EMA")

    def test_kmeans_init_improves_quantization(self):
        imgs = []
        for i in range(8):
            img = [[[0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1] for _ in range(16)] for _ in range(16)]
            imgs.append(img)

        patches1, _ = self.mm_tok.patcher.extract_patches(imgs[0])
        all_patches = patches1 * 64
        patch_vectors = [p.pixels for p in all_patches]

        self.mm_tok.codebook.kmeans_init(patch_vectors, max_iter=5)
        _, _, error = self.mm_tok.codebook.quantize_patch(patch_vectors[0])
        self.assertIsInstance(error, float)

    def test_multimodal_vocabulary_freeze(self):
        initial_vocab_size = self.mm_tok.vocab_size
        self.mm_tok.freeze()
        with self.assertRaises(KeyError):
            self.mm_tok._assign_id("<|unregistered_metadata_token|>")
        self.assertEqual(self.mm_tok.vocab_size, initial_vocab_size)

    def test_visual_codebook_finalize_rebuilds_below_100_updates(self):
        cb = VisualCodebook(num_embeddings=16, embedding_dim=8, ema_decay=0.9, epsilon=1e-5)
        init = [row[:] for row in cb.codebook]
        vectors = [[0.5] * 8]
        indices = [0]

        # Under 100 updates without finalize, codebook remains initial random vectors
        for _ in range(5):
            cb.update_ema(vectors, indices)
        self.assertEqual(cb.codebook, init)

        # finalize() flushes EMA state and rebuilds codebook vectors
        cb.finalize()
        self.assertNotEqual(cb.codebook, init)
        self.assertNotEqual(cb.codebook[0], init[0])

        # get_codebook_state() also flushes EMA state before export
        cb2 = VisualCodebook(num_embeddings=16, embedding_dim=8, ema_decay=0.9, epsilon=1e-5)
        init2 = [row[:] for row in cb2.codebook]
        for _ in range(5):
            cb2.update_ema(vectors, indices)
        state = cb2.get_codebook_state()
        self.assertNotEqual(state["codebook"], init2)
        self.assertNotEqual(cb2.codebook, init2)

    def test_visual_codebook_absent_codes_decay_each_update(self):
        cb = VisualCodebook(num_embeddings=4, embedding_dim=2, ema_decay=0.9, epsilon=1e-5)
        # Update code 0 in step 1
        cb.update_ema([[1.0, 2.0]], [0])
        size_0_after_step1 = cb._ema_cluster_size[0]
        self.assertGreater(size_0_after_step1, 0.0)

        # Update code 1 in step 2 (code 0 absent)
        cb.update_ema([[3.0, 4.0]], [1])
        size_0_after_step2 = cb._ema_cluster_size[0]
        # Absent code 0 must decay by ema_decay
        self.assertAlmostEqual(size_0_after_step2, size_0_after_step1 * 0.9, places=9)

    def test_visual_codebook_absent_code_with_zero_cluster_size_decays_embed_sum(self):
        """EMA embedding sum must decay for absent codes even when cluster size is 0."""
        state = {
            "num_embeddings": 4,
            "embedding_dim": 2,
            "seed": 42,
            "ema_decay": 0.9,
            "epsilon": 1e-5,
            "codebook": [[0.0, 0.0] for _ in range(4)],
            "ema_cluster_size": [0.0, 0.0, 0.0, 0.0],
            "ema_embed_sum": [[10.0, 20.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "update_count": 0,
        }
        cb = VisualCodebook.from_state(state)
        # Update code 1 (code 0 is absent, has cluster size 0.0 and nonzero embed sum)
        cb.update_ema([[1.0, 1.0]], [1])
        self.assertAlmostEqual(cb._ema_embed_sum[0][0], 9.0, places=9)
        self.assertAlmostEqual(cb._ema_embed_sum[0][1], 18.0, places=9)
        self.assertEqual(cb._ema_cluster_size[0], 0.0)

    def test_visual_codebook_save_and_load_roundtrip(self):
        cb = VisualCodebook(num_embeddings=8, embedding_dim=4, ema_decay=0.95, epsilon=1e-5)
        cb.update_ema([[1.0, 1.0, 1.0, 1.0]], [2])
        with TemporaryDirectory() as td:
            save_path = Path(td) / "codebook.json"
            cb.save(save_path)
            loaded = VisualCodebook.load(save_path)
            self.assertEqual(loaded.codebook, cb.codebook)
            self.assertEqual(loaded.num_embeddings, cb.num_embeddings)
            self.assertEqual(loaded.embedding_dim, cb.embedding_dim)
            self.assertEqual(loaded._update_count, cb._update_count)
            self.assertEqual(loaded._ema_cluster_size, cb._ema_cluster_size)

    def test_empty_image_element_raises_value_error(self):
        with self.assertRaises(ValueError):
            ImageElement(pixels=[])
        with self.assertRaises(ValueError):
            ImageElement(pixels=[[]])
        with self.assertRaises(TypeError):
            ImageElement(pixels="not a list")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ImageElement(pixels=[[[1.0, 2.0, 3.0]]], metadata="not a dict")  # type: ignore[arg-type]

    def test_encode_image_rejects_empty_and_invalid(self):
        with self.assertRaises(ValueError):
            self.mm_tok.encode_image([])
        with self.assertRaises(ValueError):
            self.mm_tok.encode_image([[]])
        with self.assertRaises(TypeError):
            self.mm_tok.encode_image("not a list")  # type: ignore[arg-type]

    def test_interleaved_rejects_empty_image_element(self):
        # Reproducer for Issue #15: empty ImageElement must not silently vanish
        with self.assertRaises(ValueError):
            self.mm_tok.encode_interleaved(
                [
                    TextElement("a"),
                    ImageElement(pixels=[]),
                    TextElement("b"),
                ]
            )

    def test_interleaved_with_text_element(self):
        img = [[[0.5, 0.5, 0.5] for _ in range(16)] for _ in range(16)]
        img_elem = ImageElement(pixels=img)

        seq_str = self.mm_tok.encode_interleaved(["test", img_elem, "test"])
        seq_elem = self.mm_tok.encode_interleaved([TextElement("test"), img_elem, TextElement("test")])

        self.assertEqual(seq_str.token_strings, seq_elem.token_strings)
        self.assertEqual(seq_str.token_ids, seq_elem.token_ids)
        self.assertEqual(seq_str.modality_mask, seq_elem.modality_mask)

        with self.assertRaises(TypeError):
            TextElement(text=123)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            TextElement(text="ok", metadata="not a dict")  # type: ignore[arg-type]


class TrieTests(unittest.TestCase):
    def test_prefix_trie_matches_and_accelerates_lattice(self):
        vocab = {"ab": log(0.4), "abc": log(0.6), "d": log(0.2)}
        trie = PrefixTrie.from_vocab(vocab)

        matches = trie.find_matches("abcd", 0)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0][1], "ab")
        self.assertEqual(matches[1][1], "abc")

        lat_no_trie = UnigramLattice("abcd", vocab, byte_fallback=True)
        lat_with_trie = UnigramLattice("abcd", vocab, byte_fallback=True, trie=trie)

        tokens1, score1 = lat_no_trie.viterbi()
        tokens2, score2 = lat_with_trie.viterbi()
        self.assertEqual(tokens1, tokens2)
        self.assertAlmostEqual(score1, score2)


class BPETests(unittest.TestCase):
    def test_bpe_training_and_encoding_roundtrip(self):
        corpus = ["low", "lower", "lowest", "lowering"] * 10
        trainer = BPETrainer(num_merges=10, byte_fallback=True)
        bpe_model = trainer.train(corpus)

        self.assertIn("low", bpe_model.vocab)
        encoded = bpe_model.encode("lowest")
        self.assertTrue(len(encoded) > 0)

        unk_id = bpe_model.token_to_id.get("<|unk|>", 0)
        ids = [bpe_model.token_to_id.get(t, unk_id) for t in encoded]
        decoded = bpe_model.decode(ids)
        self.assertEqual(decoded, "lowest")

    def test_bpe_heap_encode_matches_naive_algorithm(self):
        """The rank-priority heap encode must produce identical segmentations
        to the original O(merges * len) algorithm on a representative mix of
        inputs (long words, words with no learnable merges, multibyte chars,
        repeated patterns that would stress the heap's stale-entry handling).
        """
        from uniqtoken.bpe_model import BPEModel

        corpus = (
            ["the quick brown fox jumps over the lazy dog"] * 5
            + ["lowest low lower lowest lowering"] * 6
            + ["unigram tokenization byte fallback"] * 4
            + ["aaaaaa bbbbbb cccccc abcdef xyz"] * 3
        )
        trainer = BPETrainer(num_merges=20, byte_fallback=True)
        bpe_model = trainer.train(corpus)

        def naive_encode_word(model: BPEModel, word: str) -> list:
            syms = model._build_symbols(word)
            if len(syms) <= 1:
                return list(syms)
            while len(syms) > 1:
                pairs = set(zip(syms[:-1], syms[1:]))
                best = min(pairs, key=lambda p: model.merges.get(p, float("inf")))
                if best not in model.merges:
                    break
                first, second = best
                merged = first + second
                out = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == first and syms[i + 1] == second:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                syms = out
            return syms

        test_words = [
            "lowest",
            "low",
            "hello",
            "the",
            "fox",
            "lowestlowest",
            "unigram tokenization byte",
            "helloworld",
            "aaaaaa",
            "abcdef",
            "no_learned_merges_here",
            "loweringlowest",
            "a",
            "ab",
            "abcdefghij",
        ]
        for word in test_words:
            heap_result = bpe_model._encode_word(word)
            naive_result = naive_encode_word(bpe_model, word)
            self.assertEqual(
                heap_result,
                naive_result,
                f"heap vs naive disagreement on {word!r}: heap={heap_result!r} naive={naive_result!r}",
            )

    def test_bpe_heap_encode_handles_unknown_words(self):
        """Words with no learned merges (or empty merges table) must fall back
        to the per-character symbols without crashing or losing fidelity."""
        from uniqtoken.bpe_model import BPEModel

        trainer = BPETrainer(num_merges=3, byte_fallback=True)
        bpe_model = trainer.train(["hello", "world"])
        result = bpe_model._encode_word("completely_unknown_word")
        self.assertTrue(len(result) > 0)
        joined = bpe_model.decode([bpe_model.token_to_id.get(t, 0) for t in result])
        self.assertEqual(joined, "completely_unknown_word")

    def test_bpe_heap_encode_unicode_word(self):
        """A word with non-ASCII characters exercises the byte-fallback path
        (per-char to <0xNN>) combined with the heap merge loop."""
        trainer = BPETrainer(num_merges=10, byte_fallback=True)
        bpe_model = trainer.train(["café", "naïve", "résumé", "hello café"] * 4)
        result = bpe_model._encode_word("café")
        self.assertTrue(len(result) > 0)
        joined = bpe_model.decode([bpe_model.token_to_id.get(t, 0) for t in result])
        self.assertEqual(joined, "café")

    def test_bpe_stale_heap_repush_preserves_live_merges(self):
        """When a merge decrements pair counts in affected words, pairs that are
        still alive elsewhere in the corpus must not be dropped when their stale
        heap entries are popped (Issue #8)."""
        trainer = BPETrainer(target_vocab_size=300, num_merges=10)
        model = trainer.train(["ab", "abc", "bcd", "bc"])
        self.assertIn(("a", "b"), model.merges)
        self.assertIn("ab", model.vocab)
        self.assertEqual(model.encode("ab"), ["ab"])


class DecodeBatchTests(unittest.TestCase):
    def setUp(self):
        corpus = ["the quick brown fox jumps over the lazy dog"] * 4
        self.tok = CustomTokenizer.train_from_corpus(
            corpus=corpus,
            target_vocab_size=500,
            ranking_strategy="char_savings",
            verbose=False,
        )

    def test_decode_batch_round_trip(self):
        texts = [
            "the quick brown fox",
            "hello world",
            "lowest low",
            "",
            "the lazy dog",
        ]
        ids_batch = self.tok.encode_to_ids_batch(texts)
        self.assertEqual(len(ids_batch), len(texts))
        decoded = self.tok.decode_batch(ids_batch)
        self.assertEqual(len(decoded), len(texts))
        for original, decoded_text in zip(texts, decoded):
            self.assertEqual(
                original,
                decoded_text,
                f"roundtrip mismatch: {original!r} -> {decoded_text!r}",
            )

    def test_decode_batch_empty_input(self):
        self.assertEqual(self.tok.decode_batch([]), [])

    def test_decode_batch_rejects_bad_num_workers(self):
        with self.assertRaises(ValueError):
            self.tok.decode_batch([[0, 1, 2]], num_workers=0)

    def test_decode_batch_serial_and_parallel_agree(self):
        texts = ["the quick", "brown fox", "lowest"]
        ids_batch = self.tok.encode_to_ids_batch(texts)
        serial = self.tok.decode_batch(ids_batch, num_workers=1)
        parallel = self.tok.decode_batch(ids_batch, num_workers=4)
        self.assertEqual(serial, parallel)


class LatticeFastPathTests(unittest.TestCase):
    def setUp(self):
        vocab = {
            "a": log(0.1),
            "b": log(0.1),
            "c": log(0.1),
            "ab": log(0.15),
            "bc": log(0.15),
            "abc": log(0.2),
            "abcd": log(0.25),
            "xy": log(0.3),
            "xyz": log(0.2),
            "the": log(0.3),
            "ther": log(0.1),
            "\u2581": log(0.1),
            "\u2581\u2581": log(0.05),
            "<|unk|>": log(0.05),
        }
        for char in "dexyz\u0905\u0906\u0907é👍":
            vocab.setdefault(char, log(0.05))
        for b in range(256):
            b_tok = ByteFallbackEngine.byte_to_token(b)
            if b_tok not in vocab:
                vocab[b_tok] = log(0.001)
        token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        self.models = [
            UnigramModel(
                vocab=dict(vocab),
                token_to_id=dict(token_to_id),
                id_to_token={idx: tok for tok, idx in token_to_id.items()},
                special_tokens=["<|unk|>"],
                max_subword_len=8,
                byte_fallback=True,
            ),
            UnigramModel(
                vocab=dict(vocab),
                token_to_id=dict(token_to_id),
                id_to_token={idx: tok for tok, idx in token_to_id.items()},
                special_tokens=["<|unk|>"],
                max_subword_len=8,
                byte_fallback=False,
            ),
        ]

    def test_fast_path_matches_lattice_exactly(self):
        import random

        rng = random.Random(99)
        alphabet = "a b c d e x y z ▁ \u0905\u0906\u0907 é 👍".split()
        checked = 0
        for model in self.models:
            for _ in range(400):
                s = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10)))
                lattice = UnigramLattice(
                    s,
                    model.vocab,
                    max_subword_len=model.max_subword_len,
                    byte_fallback=model.byte_fallback,
                )
                lattice_tokens, _ = lattice.viterbi()
                edges, _ = lattice.viterbi_edges()
                lattice_spans = [(token, edge.start, edge.end) for edge in edges for token in edge.tokens]
                fast = model._encode_fast(s)
                self.assertEqual(model.encode(s), lattice_tokens, repr(s))
                if fast is not None:
                    self.assertEqual(
                        [t for t, _, _ in fast],
                        lattice_tokens,
                        f"fast path mismatch for {s!r}",
                    )
                self.assertEqual(
                    model.encode_with_spans(s),
                    lattice_spans,
                    f"span mismatch for {s!r}",
                )
                checked += 1
        self.assertGreater(checked, 0)


class HuggingFaceExportTests(unittest.TestCase):
    def test_hf_export_generates_valid_schema(self):
        vocab = {"tok": log(0.5), "en": log(0.3), "<|unk|>": log(0.2)}
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=["<|unk|>"],
            byte_fallback=False,
        )
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("tokenizers is not installed")

        tok = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
        with TemporaryDirectory() as tmp_dir:
            tok.export_to_huggingface(tmp_dir)
            with open(Path(tmp_dir) / "tokenizer.json", "r", encoding="utf-8") as f:
                hf_data = json.load(f)

            self.assertEqual(hf_data["version"], "1.0")
            self.assertEqual(hf_data["model"]["type"], "Unigram")
            self.assertIsNone(hf_data["post_processor"])
            self.assertTrue((Path(tmp_dir) / "tokenizer_config.json").exists())
            exported = Tokenizer.from_file(str(Path(tmp_dir) / "tokenizer.json"))
            self.assertEqual(exported.encode("token").tokens, ["tok", "en"])

    def test_hf_export_supports_byte_fallback(self):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self.skipTest("tokenizers is not installed")

        vocab = {"<|unk|>": log(1.0)}
        vocab.update({ByteFallbackEngine.byte_to_token(byte): log(0.001) for byte in range(256)})
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=["<|unk|>"],
            byte_fallback=True,
        )
        tok = CustomTokenizer(Normalizer(), RegexPreTokenizer(), model)
        hf_dict = HuggingFaceExporter.export_to_hf_dict(tok)
        self.assertTrue(hf_dict["model"]["byte_fallback"])

        with TemporaryDirectory() as tmp_dir:
            tok.export_to_huggingface(tmp_dir)
            exported = Tokenizer.from_file(str(Path(tmp_dir) / "tokenizer.json"))
            encoded = exported.encode("U0001f389")
            self.assertEqual(exported.decode(encoded.ids), "U0001f389")


class SecurityAndIndentationTests(unittest.TestCase):
    def test_security_sanitization_retains_source_alignment(self):
        shield = SecurityShield(["<|user|>"])
        sanitized, alignment = shield.sanitize_with_alignment("a<|user|>b")

        self.assertEqual(sanitized, "a<\\|user\\|>b")
        self.assertEqual(alignment[0], (0, 1))
        self.assertTrue(all(span == (1, 9) for span in alignment[1:-1]))
        self.assertEqual(alignment[-1], (9, 10))

    def test_indentation_compression_runs_during_encoding_with_offsets(self):
        vocab = {"x": log(0.5), "<|space_4|>": log(0.5)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id={"x": 0, "<|space_4|>": 1},
            id_to_token={0: "x", 1: "<|space_4|>"},
            special_tokens=["<|space_4|>"],
            byte_fallback=False,
        )
        tokenizer = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)

        self.assertEqual(tokenizer.encode("    x"), ["<|space_4|>", "x"])
        self.assertEqual(tokenizer.decode(tokenizer.encode_to_ids("    x")), "    x")
        self.assertEqual(
            [(token.text, token.raw_span) for token in tokenizer.encode_with_offsets("    x")],
            [("<|space_4|>", (0, 4)), ("x", (4, 5))],
        )
        self.assertEqual(IndentationCompressor.decompress_indents("<|space_4|>x"), "    x")

    def test_indentation_compression_skips_tokens_not_in_vocab(self):
        vocab = {"x": log(0.5), "\u2581": log(0.5)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id={"x": 0, "\u2581": 1},
            id_to_token={0: "x", 1: "\u2581"},
            special_tokens=[],
            byte_fallback=False,
        )
        tokenizer = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
        self.assertEqual(tokenizer.encode("    x"), ["\u2581", "\u2581", "\u2581", "\u2581", "x"])

    def test_rejects_invalid_security_action(self):
        shield = SecurityShield(["<|user|>"])
        with self.assertRaises(ValueError):
            shield.sanitize("<|user|>", disallowed_special_action="invalid")


class StreamingDecoderTests(unittest.TestCase):
    def test_streaming_decoder_preserves_byte_fallback_and_literal_metaspace(self):
        ids = {
            0: "<0xE2>",
            1: "<0x96>",
            2: "<0x81>",
            3: "<0xE0>",
            4: "<0x80>",
            5: "<0x81>",
            6: "<0xEE>",
        }
        decoder = StreamingDecoder(
            ids,
            metaspace_escape=("\ue000", "\ue001"),
        )
        self.assertEqual(decoder.feed_token_id(0), "")
        self.assertEqual(decoder.feed_token_id(1), "")
        self.assertEqual(decoder.feed_token_id(2), "▁")

        decoder.reset()
        escaped_tokens = [6, 4, 4, 6, 4, 5]
        self.assertEqual("".join(decoder.feed_token_id(token) for token in escaped_tokens), "▁")

    def test_streaming_decoder_keeps_valid_byte_after_invalid_prefix(self):
        decoder = StreamingDecoder({0: "<0xE2>", 1: "<0x41>"})
        self.assertEqual(decoder.feed_token_id(0), "")
        self.assertEqual(decoder.feed_token_id(1), "�A")

    def test_streaming_decoder_applies_indentation_replacements(self):
        decoder = StreamingDecoder(
            {0: "<|space_4|>"},
            special_tokens=["<|space_4|>"],
            special_replacements={"<|space_4|>": "    "},
        )
        self.assertEqual(decoder.feed_token_id(0), "    ")


class AudioCodecTests(unittest.TestCase):
    def test_audio_rvq_and_multimodal_interleaving(self):
        rvq = ResidualVectorQuantizer(num_quantizers=4, codebook_size=64, frame_size=320)
        synthetic_audio = [0.1 * math.sin(i * 0.1) for i in range(640)]

        tokens, num_frames = rvq.encode_audio(synthetic_audio)
        self.assertEqual(num_frames, 2)
        self.assertEqual(tokens[0], "<|audio_start|>")
        self.assertEqual(tokens[-1], "<|audio_end|>")

        reconstructed = rvq.decode_audio(tokens)
        self.assertEqual(len(reconstructed), len(synthetic_audio))

        vocab = {"audio": log(0.5), "<|unk|>": log(0.5)}
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=["<|unk|>"],
            byte_fallback=False,
        )
        base_tok = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
        mm_tok = MultimodalTokenizer(base_tok, patch_size=16, channels=3, num_visual_tokens=64)

        aud_segment = AudioSegment(samples=synthetic_audio)
        seq = mm_tok.encode_interleaved(["audio", aud_segment, "audio"])

        self.assertIn(0, seq.modality_mask)  # Text
        self.assertIn(2, seq.modality_mask)  # Audio
        self.assertIn(3, seq.modality_mask)  # Special

    def test_empty_audio_segment_and_encode_rejects_empty(self):
        rvq = ResidualVectorQuantizer(num_quantizers=4, codebook_size=64, frame_size=320)
        with self.assertRaises(ValueError):
            AudioSegment(samples=[])
        with self.assertRaises(ValueError):
            rvq.encode_audio([])


class NeuralCodecTests(unittest.TestCase):
    def test_neural_visual_codec_forward_and_tokens(self):
        from uniqtoken.multimodal.neural_codecs import HAS_TORCH, NeuralVisualCodec

        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        import torch

        # Batch of 2 RGB images (32x32)
        x = torch.randn(2, 3, 32, 32)
        model = NeuralVisualCodec(in_channels=3, hidden_dim=32, latent_dim=64, num_tokens=128)

        # 1. Forward training pass
        out = model(x)
        self.assertIn("loss", out)
        self.assertEqual(out["x_recon"].shape, x.shape)
        self.assertEqual(out["indices"].shape, (2, 8, 8))  # 4x downsampled

        # 2. Token emission and reconstruction
        token_strings, indices, (gh, gw) = model.encode_to_tokens(x)
        self.assertEqual(len(token_strings), 2 * 8 * 8)
        self.assertTrue(token_strings[0].startswith("<|vis_"))

        x_recon = model.decode_from_indices(indices, gh, gw)
        self.assertEqual(x_recon.shape, x.shape)

        odd_image = torch.randn(1, 3, 17, 25)
        odd_output = model(odd_image)
        self.assertEqual(odd_output["x_recon"].shape, odd_image.shape)
        _, odd_indices, (odd_gh, odd_gw) = model.encode_to_tokens(odd_image)
        self.assertEqual((odd_gh, odd_gw), (5, 7))
        self.assertEqual(
            model.decode_from_indices(odd_indices, odd_gh, odd_gw, output_size=(17, 25)).shape,
            odd_image.shape,
        )

    def test_neural_audio_codec_forward_and_tokens(self):
        from uniqtoken.multimodal.neural_codecs import HAS_TORCH, NeuralAudioCodec

        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        import torch

        # Batch of 2 audio waveforms (640 samples = 2 frames at 320 downsampling)
        audio = torch.randn(2, 1, 640)
        model = NeuralAudioCodec(
            in_channels=1,
            hidden_dim=32,
            latent_dim=64,
            num_quantizers=4,
            codebook_size=128,
        )

        # 1. Forward training pass
        out = model(audio)
        self.assertIn("loss", out)
        self.assertEqual(out["audio_recon"].shape, audio.shape)
        self.assertEqual(out["indices"].shape, (2, 2, 4))  # [B, T', N_q]

        # 2. Token emission and reconstruction
        tokens, indices = model.encode_to_tokens(audio)
        self.assertEqual(tokens[0], "<|audio_start|>")
        self.assertEqual(tokens[-1], "<|audio_end|>")

        audio_recon = model.decode_from_indices(indices)
        self.assertEqual(audio_recon.shape, audio.shape)

        short_audio = torch.randn(1, 1, 1)
        short_output = model(short_audio)
        self.assertEqual(short_output["audio_recon"].shape, short_audio.shape)
        uneven_audio = torch.randn(1, 1, 321)
        _, uneven_indices = model.encode_to_tokens(uneven_audio)
        self.assertEqual(uneven_indices.shape, (1, 2, 4))
        self.assertEqual(
            model.decode_from_indices(uneven_indices, output_length=321).shape,
            uneven_audio.shape,
        )


class CrossEntropyMergingTests(unittest.TestCase):
    def _make_pipeline(self):
        docs = [
            "the quick brown fox jumps over the lazy dog",
            "the quick fox and the lazy dog",
            "jumping foxes are quick and brown",
            "brown dogs are quick",
        ] * 20
        normalizer = Normalizer()
        pre_tokenizer = RegexPreTokenizer()
        chunks: list[str] = []
        for doc in docs:
            normalized = normalizer.normalize(doc)
            chunks.extend(pre_tokenizer.pre_tokenize(normalized))
        model = UnigramTrainer(
            target_vocab_size=300,
            max_ngram_length=6,
            min_frequency=2,
            prune_rate=0.3,
            byte_fallback=True,
        ).train(chunks, verbose=False)
        return docs, normalizer, pre_tokenizer, model, chunks

    def test_cem_reduces_corpus_token_count(self):
        _, _, _, model, chunks = self._make_pipeline()
        before = sum(len(model.encode(c)) for c in chunks)
        optimizer = CrossEntropyMerging(max_merges=100)
        improved = optimizer.optimize(model, chunks)
        after = sum(len(improved.encode(c)) for c in chunks)

        self.assertGreater(len(optimizer.merges), 0)
        self.assertLess(after, before)
        self.assertEqual(len(improved.vocab), len(model.vocab) + len(optimizer.merges))

    def test_cem_preserves_existing_token_ids(self):
        _, _, _, model, chunks = self._make_pipeline()
        optimizer = CrossEntropyMerging(max_merges=50)
        improved = optimizer.optimize(model, chunks)

        for tok, tid in model.token_to_id.items():
            self.assertEqual(improved.token_to_id[tok], tid)
        for merged in (m[2] for m in optimizer.merges):
            self.assertGreaterEqual(improved.token_to_id[merged], len(model.vocab))

    def test_cem_allocates_above_sparse_existing_ids(self):
        model = UnigramModel(
            vocab={"a": log(0.5), "b": log(0.5)},
            token_to_id={"a": 0, "b": 2},
            id_to_token={0: "a", 2: "b"},
            special_tokens=[],
            max_subword_len=2,
            byte_fallback=False,
        )
        optimizer = CrossEntropyMerging(max_merges=1)
        improved = optimizer.optimize(model, ["ab"] * 5)
        self.assertEqual(improved.token_to_id["ab"], 3)
        self.assertEqual(improved.id_to_token[2], "b")

    def test_cem_resets_merge_history_between_runs(self):
        _, _, _, model, chunks = self._make_pipeline()
        optimizer = CrossEntropyMerging(max_merges=1)
        optimizer.optimize(model, chunks)
        self.assertLessEqual(len(optimizer.merges), 1)
        optimizer.optimize(model, chunks)
        self.assertLessEqual(len(optimizer.merges), 1)

    def test_cem_roundtrip_is_lossless_through_full_pipeline(self):
        docs, normalizer, pre_tokenizer, model, chunks = self._make_pipeline()
        optimizer = CrossEntropyMerging(max_merges=50)
        improved = optimizer.optimize(model, chunks)
        tokenizer = CustomTokenizer(normalizer, pre_tokenizer, improved)

        for doc in docs:
            self.assertEqual(tokenizer.decode(tokenizer.encode_to_ids(doc)), doc)

    def test_cem_respects_max_subword_len_and_excludes_specials(self):
        _, _, _, model, chunks = self._make_pipeline()
        optimizer = CrossEntropyMerging(max_merges=100)
        optimizer.optimize(model, chunks)

        for a, b, merged, _, _ in optimizer.merges:
            self.assertLessEqual(len(merged), model.max_subword_len)
            self.assertNotIn(a, model.special_tokens)
            self.assertNotIn(b, model.special_tokens)
            self.assertNotIn(merged, model.special_tokens)
            self.assertFalse(ByteFallbackEngine.BYTE_TOKEN_PATTERN.match(a))
            self.assertFalse(ByteFallbackEngine.BYTE_TOKEN_PATTERN.match(b))

    def test_cem_max_merges_zero_returns_identical_model(self):
        _, _, _, model, chunks = self._make_pipeline()
        optimizer = CrossEntropyMerging(max_merges=0)
        self.assertIs(optimizer.optimize(model, chunks), model)
        with self.assertRaises(ValueError):
            CrossEntropyMerging(max_merges=-1)


class SuperBPETests(unittest.TestCase):
    SPACE = "\u2581"

    def _make_pipeline(self):
        docs = [
            "the quick brown fox jumps over the lazy dog",
            "the quick fox and the lazy dog",
            "jumping foxes are quick and brown",
            "brown dogs are quick",
        ] * 30
        normalizer = Normalizer()
        pre_tokenizer = RegexPreTokenizer()
        chunks: list[str] = []
        for doc in docs:
            normalized = normalizer.normalize(doc)
            chunks.extend(pre_tokenizer.pre_tokenize(normalized))
        base_model = UnigramTrainer(
            target_vocab_size=300,
            max_ngram_length=6,
            min_frequency=2,
            prune_rate=0.3,
            byte_fallback=True,
        ).train(chunks, verbose=False)
        optimizer = CrossEntropyMerging(max_merges=100, cross_word=True)
        improved = optimizer.optimize(base_model, chunks)
        base_tok = CustomTokenizer(normalizer, pre_tokenizer, base_model)
        improved_tok = CustomTokenizer(normalizer, pre_tokenizer, improved)
        return docs, normalizer, pre_tokenizer, base_tok, improved_tok, optimizer

    def test_superbpe_merges_span_word_boundaries(self):
        _, _, _, _, _, optimizer = self._make_pipeline()
        self.assertGreater(len(optimizer.merges), 0)
        for a, b, merged, _, _ in optimizer.merges:
            self.assertIn(self.SPACE, merged)
            self.assertNotIn(a, ("<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"))
            self.assertNotIn(b, ("<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"))

    def test_superbpe_replays_hierarchical_merges(self):
        vocab = {
            "a": log(0.2),
            self.SPACE: log(0.2),
            "b": log(0.2),
            "a" + self.SPACE: log(0.2),
            "a" + self.SPACE + "b": log(0.2),
        }
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=[],
            max_subword_len=3,
            byte_fallback=False,
        )
        tokenizer = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
        self.assertEqual(tokenizer.encode("a b"), ["a" + self.SPACE + "b"])
        token = tokenizer.encode_with_offsets("a b")[0]
        self.assertEqual(token.raw_span, (0, 3))

    def test_superbpe_reduces_token_count_through_pipeline(self):
        docs, _, _, base_tok, improved_tok, _ = self._make_pipeline()
        full_text = " ".join(docs)
        before = len(base_tok.encode(full_text))
        after = len(improved_tok.encode(full_text))

        self.assertLess(after, before)
        self.assertTrue(any(tok in improved_tok.encode(full_text) for tok in ["the" + self.SPACE, self.SPACE + "fox"]))

    def test_superbpe_roundtrip_lossless_through_pipeline(self):
        docs, _, _, _, improved_tok, _ = self._make_pipeline()
        for doc in docs:
            self.assertEqual(improved_tok.decode(improved_tok.encode_to_ids(doc)), doc)

    def test_superbpe_preserves_ids_and_grows_vocab(self):
        _, _, _, base_tok, improved_tok, optimizer = self._make_pipeline()
        for tok, tid in base_tok.model.token_to_id.items():
            self.assertEqual(improved_tok.model.token_to_id[tok], tid)
        self.assertEqual(
            len(improved_tok.model.vocab),
            len(base_tok.model.vocab) + len(optimizer.merges),
        )
        for merged in (m[2] for m in optimizer.merges):
            self.assertGreaterEqual(improved_tok.model.token_to_id[merged], len(base_tok.model.vocab))

    def test_superbpe_respects_max_subword_len(self):
        _, _, _, base_tok, _, optimizer = self._make_pipeline()
        for a, b, merged, _, _ in optimizer.merges:
            self.assertLessEqual(len(merged), base_tok.model.max_subword_len)
            self.assertFalse(ByteFallbackEngine.BYTE_TOKEN_PATTERN.match(a))
            self.assertFalse(ByteFallbackEngine.BYTE_TOKEN_PATTERN.match(b))

    def test_superbpe_offset_spans_are_contiguous(self):
        _, _, _, _, improved_tok, _ = self._make_pipeline()
        for doc in ["the quick brown fox", "brown dogs are quick"]:
            tokens = improved_tok.encode_with_offsets(doc)
            self.assertEqual(tokens[0].raw_span[0], 0)
            self.assertEqual(tokens[-1].raw_span[1], len(doc))
            prev_end = tokens[0].raw_span[0]
            for tok in tokens:
                start, end = tok.raw_span
                self.assertEqual(start, prev_end)
                prev_end = end
            self.assertEqual(improved_tok.decode([t.id for t in tokens]), doc)

    def test_superbpe_sample_applies_cross_word_merges(self):
        _, _, _, _, improved_tok, _ = self._make_pipeline()
        sample_tokens = improved_tok.sample("the quick brown fox", alpha=0.5)
        self.assertTrue(len(sample_tokens) > 0)
        sample_ids = [
            improved_tok.model.token_to_id.get(t, improved_tok.model.token_to_id.get("<|unk|>", 0))
            for t in sample_tokens
        ]
        self.assertEqual(improved_tok.decode(sample_ids), "the quick brown fox")

    def test_superbpe_model_reassignment_invalidates_cross_word_cache(self):
        _, _, _, base_tok, improved_tok, _ = self._make_pipeline()
        # Warm up cache on base_tok (no cross-word tokens yet)
        base_cw = base_tok._cross_word_tokens()
        self.assertEqual(len(base_cw), 0)
        # Reassign model in place to improved_tok.model (has cross-word tokens)
        base_tok.model = improved_tok.model
        reassigned_cw = base_tok._cross_word_tokens()
        self.assertGreater(len(reassigned_cw), 0)

    def test_superbpe_leading_metaspace_cross_word_tokens_recognized_and_emitted(self):
        """Cross-word tokens with a leading metaspace (e.g. "\u2581over\u2581the")
        must be recognized by _cross_word_tokens() and emitted by encode() (Issue #9)."""
        vocab = {
            "over": log(0.25),
            self.SPACE + "the": log(0.25),
            self.SPACE + "over": log(0.25),
            self.SPACE + "over" + self.SPACE + "the": log(0.25),
        }
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=[],
            max_subword_len=12,
            byte_fallback=False,
        )
        tokenizer = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
        leading_cw = self.SPACE + "over" + self.SPACE + "the"
        self.assertIn(leading_cw, tokenizer._cross_word_tokens())

        # When encoding " over the", the two chunks are "\u2581over" and "\u2581the",
        # which must be fused into "\u2581over\u2581the" by _apply_cross_word_merges.
        encoded = tokenizer.encode(" over the")
        self.assertEqual(encoded, [leading_cw])

    def test_superbpe_cem_does_not_create_intra_word_dead_merges(self):
        """CEM in cross_word mode must only accept pairs whose concatenation has
        an internal metaspace (at index > 0), avoiding dead intra-word merges (Issue #9)."""
        vocab = {
            self.SPACE: log(0.25),
            "quick": log(0.25),
            self.SPACE + "fox": log(0.25),
            "other": log(0.25),
        }
        token_to_id = {token: index for index, token in enumerate(vocab)}
        model = UnigramModel(
            vocab=vocab,
            token_to_id=token_to_id,
            id_to_token={index: token for token, index in token_to_id.items()},
            special_tokens=[],
            max_subword_len=16,
            byte_fallback=False,
        )
        optimizer = CrossEntropyMerging(max_merges=10, cross_word=True)
        # "\u2581quick" encodes as ["\u2581", "quick"], generating the candidate pair
        # ("\u2581", "quick") alongside the valid cross-word candidate ("quick", "\u2581fox").
        chunks = [self.SPACE + "quick", self.SPACE + "fox"] * 20
        improved = optimizer.optimize(model, chunks)
        tokenizer = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), improved)
        self.assertGreater(len(optimizer.merges), 0, "Expected at least one valid cross-word merge")
        for _, _, merged, _, _ in optimizer.merges:
            self.assertIn(self.SPACE, merged[1:], f"Merge {merged!r} should have an internal metaspace")
            self.assertIn(merged, tokenizer._cross_word_tokens(), f"Merge {merged!r} must not be dead")

    def test_image_patcher_empty_nested_pixels(self):
        from uniqtoken.multimodal.image_patcher import DynamicImagePatcher

        patcher = DynamicImagePatcher(patch_size=4, channels=3)
        self.assertEqual(patcher.extract_patches([]), ([], (0, 0)))
        self.assertEqual(patcher.extract_patches([[]]), ([], (0, 0)))

    def test_vocab_adapter_extreme_underflow(self):
        from uniqtoken.vocab_adapter import VocabularyAdapter

        _, _, _, base_tok, _, _ = self._make_pipeline()
        # Simulate extreme negative log-probs
        base_tok.model.vocab["rare_token"] = -1000.0
        adapted = VocabularyAdapter.expand_vocabulary(
            base_tok, ["xyzabc xyzabc xyzabc"], num_new_tokens=5, min_frequency=1, verbose=False
        )
        self.assertGreater(len(adapted.model.vocab), len(base_tok.model.vocab))
        for tok, lp in adapted.model.vocab.items():
            self.assertFalse(math.isnan(lp))
            self.assertFalse(math.isinf(lp))


class PhaseOneOptimizationTests(unittest.TestCase):
    def test_seed_builder_pmi_ranking_and_adaptive_sizing(self):
        builder = SeedVocabularyBuilder(
            target_vocab_size=300,
            seed_multiplier=2.0,
            ranking_strategy="pmi",
            adaptive_multiplier=True,
            min_frequency=1,
        )
        chunks = ["neural", "network", "language", "model", "neural", "model", "transformer"]
        seed_vocab = builder.build_seed_vocab(chunks)
        self.assertGreater(len(seed_vocab), 0)
        tokens = [t.token for t in seed_vocab]
        self.assertIn("model", tokens)

        # Invalid ranking strategy check
        with self.assertRaises(ValueError):
            SeedVocabularyBuilder(ranking_strategy="invalid_strategy")

    def test_lattice_beam_pruning_and_min_edge_threshold(self):
        vocab = {
            "a": log(0.3),
            "b": log(0.3),
            "ab": log(0.2),
            "c": log(0.1),
            "abc": log(0.05),
            "rare": log(1e-6),
        }
        # Beam pruning: max 1 incoming edge per node
        lattice = UnigramLattice(
            "abc",
            vocab,
            max_subword_len=3,
            byte_fallback=True,
            max_edges_per_node=1,
            min_edge_log_prob=log(0.01),
        )
        for j in range(1, len("abc") + 1):
            self.assertLessEqual(len(lattice.end_nodes[j]), 1)

        # Invalid max_edges_per_node check
        with self.assertRaises(ValueError):
            UnigramLattice("abc", vocab, max_edges_per_node=0)

    def test_unigram_trainer_convergence_early_stopping(self):
        corpus = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps",
            "brown fox jumps over",
        ]
        tok = CustomTokenizer.train_from_corpus(
            corpus,
            target_vocab_size=320,
            ranking_strategy="pmi",
            adaptive_multiplier=True,
            max_edges_per_node=5,
            convergence_tolerance=1e-3,
            verbose=False,
        )
        self.assertIsInstance(tok, CustomTokenizer)
        encoded = tok.encode("the quick brown fox")
        self.assertTrue(len(encoded) > 0)
        decoded = tok.decode(tok.encode_to_ids("the quick brown fox"))
        self.assertEqual(decoded, "the quick brown fox")

    def test_encode_with_metrics_diagnostic_report(self):
        corpus = ["standard test sentence with normal alphabet"]
        tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=320, verbose=False)
        report = tok.encode_with_metrics("standard test sentence 🚀")
        self.assertIsInstance(report, TokenizationReport)
        self.assertGreater(report.num_tokens, 0)
        self.assertGreater(report.num_bytes, 0)
        self.assertGreater(report.num_chars, 0)
        # Byte fallback should catch emoji 🚀 if not in alphabet
        self.assertGreaterEqual(report.byte_fallback_tokens, 0)
        self.assertGreaterEqual(report.byte_fallback_rate, 0.0)
        self.assertLessEqual(report.byte_fallback_rate, 1.0)
        self.assertGreater(report.compression_ratio_bytes_per_token, 0.0)

    def test_complex_indic_and_arabic_unicode_offsets(self):
        corpus = [
            "प्राकृतिक भाषा प्रसंस्करण नमस्ते दुनिया",
            "تعتبر معالجة اللغات الطبيعية الحديثة",
        ]
        tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=350, verbose=False)

        for text in corpus:
            tokens = tok.encode_with_offsets(text)
            self.assertTrue(len(tokens) > 0)
            self.assertEqual(tokens[0].raw_span[0], 0)
            self.assertEqual(tokens[-1].raw_span[1], len(text))
            for t in tokens:
                self.assertGreaterEqual(t.raw_span[0], 0)
                self.assertLessEqual(t.raw_span[1], len(text))
                self.assertLessEqual(t.raw_span[0], t.raw_span[1])
            # Monotonic start span progression
            for i in range(len(tokens) - 1):
                self.assertLessEqual(tokens[i].raw_span[0], tokens[i + 1].raw_span[0])
            # Lossless roundtrip
            decoded = tok.decode([t.id for t in tokens])
            self.assertEqual(decoded, text)


class PhaseTwoOptimizationTests(unittest.TestCase):
    def test_batch_encoding_and_offsets(self):
        corpus = [
            "the quick brown fox jumps over the lazy dog",
            "fast parallel batch encoding verification",
            "neural language model subword regularization",
        ]
        tok = CustomTokenizer.train_from_corpus(corpus, target_vocab_size=320, verbose=False)

        # Batch encode vs sequential
        seq_tokens = [tok.encode(t) for t in corpus]
        batch_tokens = tok.encode_batch(corpus, num_workers=2)
        self.assertEqual(seq_tokens, batch_tokens)

        # Batch IDs vs sequential
        seq_ids = [tok.encode_to_ids(t) for t in corpus]
        batch_ids = tok.encode_to_ids_batch(corpus, num_workers=2)
        self.assertEqual(seq_ids, batch_ids)

        # Batch offsets vs sequential
        seq_offsets = [[tok_obj.raw_span for tok_obj in tok.encode_with_offsets(t)] for t in corpus]
        batch_offsets = [
            [tok_obj.raw_span for tok_obj in batch] for batch in tok.encode_with_offsets_batch(corpus, num_workers=2)
        ]
        self.assertEqual(seq_offsets, batch_offsets)

    def test_compact_trie_slots_and_id_mapping(self):
        from uniqtoken.trie import PrefixTrie, TrieNode

        node = TrieNode()
        self.assertTrue(hasattr(node, "__slots__"))
        self.assertFalse(hasattr(node, "__dict__"))

        vocab = {"hello": log(0.5), "world": log(0.5)}
        token_to_id = {"hello": 101, "world": 102}
        trie = PrefixTrie.from_vocab(vocab, token_to_id=token_to_id)
        matches = trie.find_matches("helloworld", 0)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][1], "hello")

    def test_benchmark_vocab_scaling_and_exporters(self):
        from benchmarks.benchmark_suite import TokenizerBenchmarkSuite

        suite = TokenizerBenchmarkSuite()
        scaling_results = suite.evaluate_vocab_scaling(vocab_sizes=[550])
        self.assertEqual(len(scaling_results), 1)
        self.assertIn("tokens", scaling_results[0])

        with TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "report.md"
            latex_path = Path(tmp_dir) / "report.tex"
            suite.export_markdown_report(str(md_path))
            suite.export_latex_report(str(latex_path))
            self.assertTrue(md_path.exists())
            self.assertTrue(latex_path.exists())
            self.assertGreater(md_path.stat().st_size, 0)
            self.assertGreater(latex_path.stat().st_size, 0)

    def test_downstream_evaluator_smoke(self):
        from benchmarks.downstream_eval import DownstreamEvaluator

        corpus = [
            "the quick brown fox jumps over the lazy dog",
            "neural language model downstream transformer evaluation",
        ]
        evaluator = DownstreamEvaluator(vocab_size=300, max_merges=5, corpus=corpus)
        results = evaluator.run_downstream_suite(include_external_baselines=False)
        self.assertGreaterEqual(len(results), 2)
        for r in results:
            self.assertGreater(r.total_tokens, 0)
            self.assertGreater(r.bytes_per_token, 0.0)
            self.assertGreater(r.effective_bytes_in_2k_context, 0)


class GGUFExportTests(unittest.TestCase):
    """Unit tests for GGUF tokenizer metadata export and score extraction."""

    def setUp(self):
        """Sets up synthetic UnigramModel and CustomTokenizer fixture with diverse token types."""
        self.vocab = {
            "<|unk|>": -10.0,
            "<|bos|>": -10.0,
            "<|eos|>": -10.0,
            "<|pad|>": -10.0,
            "<0x41>": -5.5,
            "<0x0A>": -6.25,
            "hello": math.log(0.4),
            "world": math.log(0.3),
            "uniq": -2.71828,
            "token": -1.41421,
        }
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.model = UnigramModel(
            vocab=self.vocab,
            token_to_id=self.token_to_id,
            id_to_token=self.id_to_token,
            special_tokens=["<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>"],
            max_subword_len=5,
            byte_fallback=True,
            unk_token="<|unk|>",
        )
        self.tokenizer = CustomTokenizer(
            normalizer=Normalizer(),
            pre_tokenizer=RegexPreTokenizer(),
            model=self.model,
        )

    def test_gguf_dict_generation(self):
        """Verifies GGUF dictionary schema keys, ordered token sequence, and token types."""
        meta = HuggingFaceExporter.export_to_gguf_dict(self.tokenizer)
        self.assertEqual(meta["tokenizer.ggml.model"], "llama")

        tokens = meta["tokenizer.ggml.tokens"]
        scores = meta["tokenizer.ggml.scores"]
        token_types = meta["tokenizer.ggml.token_type"]

        self.assertEqual(len(tokens), len(self.vocab))
        self.assertEqual(len(scores), len(self.vocab))
        self.assertEqual(len(token_types), len(self.vocab))

        # Contiguous tokens in ID order
        for idx, tok in enumerate(tokens):
            self.assertEqual(tok, self.id_to_token[idx])
            self.assertAlmostEqual(scores[idx], self.vocab[tok], places=5)

        # Verify token types
        token_type_map = dict(zip(tokens, token_types))
        self.assertEqual(token_type_map["<|unk|>"], GGUFTokenType.UNKNOWN)
        self.assertEqual(token_type_map["<|bos|>"], GGUFTokenType.CONTROL)
        self.assertEqual(token_type_map["<|eos|>"], GGUFTokenType.CONTROL)
        self.assertEqual(token_type_map["<|pad|>"], GGUFTokenType.CONTROL)
        self.assertEqual(token_type_map["<0x41>"], GGUFTokenType.BYTE)
        self.assertEqual(token_type_map["<0x0A>"], GGUFTokenType.BYTE)
        self.assertEqual(token_type_map["hello"], GGUFTokenType.NORMAL)
        self.assertEqual(token_type_map["world"], GGUFTokenType.NORMAL)

        # Verify special token IDs
        self.assertEqual(meta["tokenizer.ggml.bos_token_id"], self.token_to_id["<|bos|>"])
        self.assertEqual(meta["tokenizer.ggml.eos_token_id"], self.token_to_id["<|eos|>"])
        self.assertEqual(meta["tokenizer.ggml.unknown_token_id"], self.token_to_id["<|unk|>"])
        self.assertEqual(meta["tokenizer.ggml.padding_token_id"], self.token_to_id["<|pad|>"])

    def test_gguf_round_trip_score_extraction(self):
        """Verifies round-trip extraction of scores from both memory bytes and saved files."""
        # 1. Round-trip in-memory bytes
        gguf_bytes = self.tokenizer.export_to_gguf()
        self.assertIsInstance(gguf_bytes, bytes)
        self.assertGreater(len(gguf_bytes), 24)

        extracted_scores = extract_gguf_scores(gguf_bytes)
        self.assertEqual(len(extracted_scores), len(self.vocab))

        for tok, orig_score in self.vocab.items():
            self.assertIn(tok, extracted_scores)
            extracted_score = extracted_scores[tok]
            self.assertAlmostEqual(
                extracted_score,
                orig_score,
                places=5,
                msg=f"Score mismatch for token {tok!r}: expected {orig_score}, got {extracted_score}",
            )

        # 2. Round-trip file persistence
        with TemporaryDirectory() as tmp_dir:
            gguf_path = Path(tmp_dir) / "test_model.gguf"
            HuggingFaceExporter.save_gguf(self.tokenizer, gguf_path)
            self.assertTrue(gguf_path.exists())
            self.assertGreater(gguf_path.stat().st_size, 24)

            file_scores = extract_gguf_scores(gguf_path)
            self.assertEqual(len(file_scores), len(self.vocab))
            for tok, orig_score in self.vocab.items():
                self.assertAlmostEqual(file_scores[tok], orig_score, places=5)

            # Also verify full metadata extraction from file
            metadata = extract_gguf_metadata(gguf_path)
            self.assertEqual(metadata["tokenizer.ggml.model"], "llama")
            self.assertEqual(metadata["tokenizer.ggml.tokens"], list(self.token_to_id.keys()))
            self.assertEqual(metadata["tokenizer.ggml.unknown_token_id"], self.token_to_id["<|unk|>"])

    def test_gguf_non_contiguous_ids_rejected(self):
        """Verifies that non-contiguous token IDs starting from 0 raise ValueError."""
        sparse_token_to_id = {"a": 0, "b": 2}
        sparse_model = UnigramModel(
            vocab={"a": -1.0, "b": -2.0},
            token_to_id=sparse_token_to_id,
            id_to_token={0: "a", 2: "b"},
            special_tokens=[],
            byte_fallback=False,
        )
        sparse_tok = CustomTokenizer(Normalizer(), RegexPreTokenizer(), sparse_model)
        with self.assertRaises(ValueError):
            HuggingFaceExporter.export_to_gguf_dict(sparse_tok)

    def test_gguf_user_defined_and_fallback_scores(self):
        """Verifies classification of user-defined special tokens and default fallback scores."""
        vocab = {"<|user_flag|>": -0.5, "unscored_special": -10.0}
        # Note: 'unscored_special' is NOT in model.vocab, will trigger default score -10.0
        model_vocab = {"<|user_flag|>": -0.5}
        token_to_id = {"<|user_flag|>": 0, "unscored_special": 1}
        model = UnigramModel(
            vocab=model_vocab,
            token_to_id=token_to_id,
            id_to_token={0: "<|user_flag|>", 1: "unscored_special"},
            special_tokens=["<|user_flag|>", "unscored_special"],
            byte_fallback=False,
        )
        tok = CustomTokenizer(Normalizer(), RegexPreTokenizer(), model)
        meta = GGUFExporter.export_to_gguf_dict(tok)

        self.assertEqual(meta["tokenizer.ggml.token_type"][0], GGUFTokenType.USER_DEFINED)
        self.assertEqual(meta["tokenizer.ggml.token_type"][1], GGUFTokenType.CONTROL)
        self.assertEqual(meta["tokenizer.ggml.scores"][1], -10.0)

        # Round-trip via GGUFExporter alias
        gguf_bytes = GGUFExporter.export_to_gguf(tok)
        scores = extract_gguf_scores(gguf_bytes)
        self.assertAlmostEqual(scores["<|user_flag|>"], -0.5, places=5)
        self.assertAlmostEqual(scores["unscored_special"], -10.0, places=5)

    def test_gguf_invalid_binary_rejected(self):
        """Verifies that corrupted magic, bad version, or truncated KV payloads raise ValueError."""
        with self.assertRaises(ValueError):
            extract_gguf_metadata(b"too_short")

        with self.assertRaises(ValueError):
            extract_gguf_metadata(b"NOTGGUF" + b"\x00" * 30)

        with self.assertRaises(ValueError):
            extract_gguf_metadata(b"GGUF" + struct.pack("<IQQ", 999, 0, 0))

        # Truncated KV section raises ValueError (not raw struct.error)
        truncated_kv = b"GGUF" + struct.pack("<IQQ", 3, 0, 5) + b"\x00" * 8
        with self.assertRaises(ValueError):
            extract_gguf_metadata(truncated_kv)

    def test_gguf_duplicate_tokens_rejected(self):
        """Verifies that duplicate tokens in GGUF vocabulary tables raise ValueError."""
        from uniqtoken.hf_exporter import GGUFValueType

        data = bytearray()
        data.extend(b"GGUF")
        data.extend(struct.pack("<IQQ", 3, 0, 2))

        def pack_str(s: str) -> bytes:
            """Packs length-prefixed string bytes."""
            b = s.encode("utf-8")
            return struct.pack("<Q", len(b)) + b

        # Key 1: tokenizer.ggml.tokens (duplicate 'hello')
        data.extend(pack_str("tokenizer.ggml.tokens"))
        data.extend(struct.pack("<IIQ", GGUFValueType.ARRAY, GGUFValueType.STRING, 2))
        data.extend(pack_str("hello"))
        data.extend(pack_str("hello"))

        # Key 2: tokenizer.ggml.scores
        data.extend(pack_str("tokenizer.ggml.scores"))
        data.extend(struct.pack("<IIQ", GGUFValueType.ARRAY, GGUFValueType.FLOAT32, 2))
        data.extend(struct.pack("<2f", -1.0, -2.0))

        with self.assertRaises(ValueError):
            extract_gguf_scores(bytes(data))

    def test_gguf_non_list_tokens_or_scores_rejected(self):
        """Verifies that non-list values for tokens or scores in GGUF metadata raise ValueError."""
        from uniqtoken.hf_exporter import GGUFValueType

        data = bytearray()
        data.extend(b"GGUF")
        data.extend(struct.pack("<IQQ", 3, 0, 2))

        def pack_str(s: str) -> bytes:
            """Packs length-prefixed string bytes."""
            b = s.encode("utf-8")
            return struct.pack("<Q", len(b)) + b

        # Key 1: tokenizer.ggml.tokens as scalar string instead of array
        data.extend(pack_str("tokenizer.ggml.tokens"))
        data.extend(struct.pack("<I", GGUFValueType.STRING))
        data.extend(pack_str("hello"))

        # Key 2: tokenizer.ggml.scores as float array
        data.extend(pack_str("tokenizer.ggml.scores"))
        data.extend(struct.pack("<IIQ", GGUFValueType.ARRAY, GGUFValueType.FLOAT32, 1))
        data.extend(struct.pack("<f", -1.0))

        with self.assertRaises(ValueError):
            extract_gguf_scores(bytes(data))

    def test_gguf_non_string_tokens_or_non_float_scores_rejected(self):
        """Verifies that non-string tokens or non-float scores raise ValueError."""
        from uniqtoken.hf_exporter import GGUFValueType

        data = bytearray()
        data.extend(b"GGUF")
        data.extend(struct.pack("<IQQ", 3, 0, 2))

        def pack_str(s: str) -> bytes:
            b = s.encode("utf-8")
            return struct.pack("<Q", len(b)) + b

        # Key 1: tokenizer.ggml.tokens as int array instead of string array
        data.extend(pack_str("tokenizer.ggml.tokens"))
        data.extend(struct.pack("<IIQ", GGUFValueType.ARRAY, GGUFValueType.INT32, 2))
        data.extend(struct.pack("<2i", 1, 2))

        # Key 2: tokenizer.ggml.scores as float array
        data.extend(pack_str("tokenizer.ggml.scores"))
        data.extend(struct.pack("<IIQ", GGUFValueType.ARRAY, GGUFValueType.FLOAT32, 2))
        data.extend(struct.pack("<2f", -1.0, -2.0))

        with self.assertRaises(ValueError):
            extract_gguf_scores(bytes(data))


if __name__ == "__main__":
    unittest.main()
