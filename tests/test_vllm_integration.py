"""Integration tests for vLLM Custom Tokenizer and Detokenizer Adapter (Issue #27).

Tests:
1. Adapter properties, vocab conversion, and serialization.
2. Synchronous & asynchronous batch encoding/decoding parity.
3. Mock vLLM incremental detokenizer consumption.
4. UTF-8 multi-byte buffering during incremental streaming without U+FFFD.
5. Stop string detection and stream termination.
6. Highly concurrent multi-threaded streaming detokenization.
7. Asynchronous streaming worker consuming token queues.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from math import log
import os
from tempfile import TemporaryDirectory
import unittest

from uniqtoken.integrations.vllm import (
    AsyncVLLMStreamingWorker,
    UniqTokenVLLMAdapter,
    VLLMDetokenizer,
    VLLMStreamingState,
)
from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.streaming_decoder import StreamingDecoder
from uniqtoken.tokenizer import CustomTokenizer
from uniqtoken.unigram_trainer import UnigramModel


def _build_test_tokenizer() -> CustomTokenizer:
    """Creates a trained UniqToken tokenizer for integration testing."""
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test sentence for vLLM integration.",
        "Stop here. Non-blocking streaming detokenization under concurrent worker load.",
    ]
    return CustomTokenizer.train_from_corpus(
        corpus=corpus,
        target_vocab_size=350,
        special_tokens=["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"],
        verbose=False,
    )


class VLLMIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _build_test_tokenizer()
        self.adapter = UniqTokenVLLMAdapter(self.tokenizer)

    def test_adapter_initialization_and_properties(self) -> None:
        self.assertTrue(self.adapter.is_fast)
        self.assertEqual(self.adapter.vocab_size, len(self.tokenizer.model.vocab))
        self.assertEqual(len(self.adapter), self.adapter.vocab_size)

        self.assertEqual(self.adapter.pad_token, "<|pad|>")
        self.assertEqual(self.adapter.eos_token, "<|eos|>")
        self.assertEqual(self.adapter.bos_token, "<|bos|>")
        self.assertEqual(self.adapter.unk_token, "<|unk|>")

        self.assertIsNotNone(self.adapter.pad_token_id)
        self.assertIsNotNone(self.adapter.eos_token_id)
        self.assertIsNotNone(self.adapter.bos_token_id)
        self.assertIsNotNone(self.adapter.unk_token_id)

        # Vocab dictionary
        vocab = self.adapter.get_vocab()
        self.assertEqual(len(vocab), self.adapter.vocab_size)

        # ID <-> Token conversion
        sample_token = next(iter(vocab))
        sample_id = self.adapter.convert_tokens_to_ids(sample_token)
        self.assertIsInstance(sample_id, int)
        self.assertEqual(self.adapter.convert_ids_to_tokens(sample_id), sample_token)

        batch_tokens = list(vocab.keys())[:5]
        batch_ids = self.adapter.convert_tokens_to_ids(batch_tokens)
        self.assertEqual(len(batch_ids), len(batch_tokens))
        self.assertEqual(self.adapter.convert_ids_to_tokens(batch_ids), batch_tokens)

    def test_save_and_from_pretrained_roundtrip(self) -> None:
        with TemporaryDirectory() as td:
            self.tokenizer.save(td)
            loaded_adapter = UniqTokenVLLMAdapter.from_pretrained(td)
            self.assertEqual(loaded_adapter.vocab_size, self.adapter.vocab_size)
            self.assertEqual(loaded_adapter.pad_token_id, self.adapter.pad_token_id)

    def test_sync_and_async_batch_encoding(self) -> None:
        prompts = [
            "The quick brown fox",
            "jumps over the lazy dog.",
            "Hello world",
        ]
        # Sync encoding
        sync_batch = self.adapter.encode_batch(prompts)
        self.assertEqual(len(sync_batch), 3)
        for prompt, seq in zip(prompts, sync_batch):
            self.assertEqual(self.adapter.encode(prompt), seq)

        # Async batch encoding
        async def run_async_encode():
            return await self.adapter.encode_batch_async(prompts)

        async_batch = asyncio.run(run_async_encode())
        self.assertEqual(sync_batch, async_batch)

        # With special tokens
        sync_special = self.adapter.encode_batch(prompts, add_special_tokens=True)
        for seq in sync_special:
            self.assertEqual(seq[0], self.adapter.bos_token_id)
            self.assertEqual(seq[-1], self.adapter.eos_token_id)

    def test_sync_and_async_batch_decoding(self) -> None:
        prompts = [
            "The quick brown fox",
            "Hello world",
        ]
        batch_ids = self.adapter.encode_batch(prompts)

        # Sync decode
        decoded_sync = self.adapter.decode_batch(batch_ids)
        for prompt, dec in zip(prompts, decoded_sync):
            self.assertEqual(dec.strip(), prompt.strip())

        # Async decode
        async def run_async_decode():
            return await self.adapter.decode_batch_async(batch_ids)

        decoded_async = asyncio.run(run_async_decode())
        self.assertEqual(decoded_sync, decoded_async)

    def test_mock_vllm_detokenizer_consumption(self) -> None:
        """Demonstrates realistic step-by-step token consumption by a mock vLLM Detokenizer."""
        detokenizer = VLLMDetokenizer(self.adapter)
        prompt = "The quick brown fox jumps over the lazy dog."
        token_ids = self.adapter.encode(prompt)

        req_id = "request-001"
        detokenizer.init_request(req_id)

        streamed_chunks: list[str] = []
        for tid in token_ids:
            delta, is_finished = detokenizer.step(req_id, tid)
            if delta:
                streamed_chunks.append(delta)
            self.assertFalse(is_finished)

        # Final flush at sequence completion
        final_flush = detokenizer.flush(req_id)
        if final_flush:
            streamed_chunks.append(final_flush)

        accumulated_text = "".join(streamed_chunks)
        direct_decoded = self.adapter.decode(token_ids)
        self.assertEqual(accumulated_text, direct_decoded)
        self.assertFalse(self.adapter.has_streaming_request(req_id))

    def test_vllm_detokenize_incrementally_protocol(self) -> None:
        prompt = "Hello world"
        token_ids = self.adapter.encode(prompt)
        req_id = "req_inc_1"

        deltas = []
        for tid in token_ids:
            delta = self.adapter.detokenize_incrementally(req_id, tid)
            if delta:
                deltas.append(delta)

        final_delta = self.adapter.flush_streaming(req_id)
        if final_delta:
            deltas.append(final_delta)

        self.assertEqual("".join(deltas), "Hello world")

    def test_utf8_multi_byte_buffering_without_glitches(self) -> None:
        """
        Verify that streaming UTF-8 bytes (split into byte fallback tokens) buffers
        correctly without emitting replacement character (U+FFFD).
        """
        # Euro sign '€' in UTF-8 is 3 bytes: 0xE2 0x82 0xAC
        b1 = self.adapter.tokenizer.model.token_to_id["<0xE2>"]
        b2 = self.adapter.tokenizer.model.token_to_id["<0x82>"]
        b3 = self.adapter.tokenizer.model.token_to_id["<0xAC>"]

        req_id = "euro_req"
        self.adapter.init_streaming_request(req_id)

        # First byte fed: should return "" and NOT \ufffd
        delta1 = self.adapter.step_streaming(req_id, b1)[0]
        self.assertEqual(delta1, "")

        # Second byte fed: should return "" and NOT \ufffd
        delta2 = self.adapter.step_streaming(req_id, b2)[0]
        self.assertEqual(delta2, "")

        # Third byte fed: completed '€' emitted immediately!
        delta3 = self.adapter.step_streaming(req_id, b3)[0]
        self.assertEqual(delta3, "€")

        flush = self.adapter.flush_streaming(req_id)
        self.assertEqual(flush, "")

    def test_stop_string_termination(self) -> None:
        """Verify that streaming halts and truncates on stop string."""
        detokenizer = VLLMDetokenizer(self.adapter)
        req_id = "stop_req"
        detokenizer.init_request(req_id, stop_strings=["Stop"])

        # Feed tokens for: "Hello world Stop here"
        tokens = self.adapter.encode("Hello world Stop here")

        deltas: list[str] = []
        is_stopped = False
        for tid in tokens:
            delta, is_finished = detokenizer.step(req_id, tid)
            if delta:
                deltas.append(delta)
            if is_finished:
                is_stopped = True
                break

        self.assertTrue(is_stopped)
        self.assertEqual("".join(deltas).strip(), "Hello world")

    def test_concurrent_multithreaded_streaming(self) -> None:
        """50 concurrent worker threads streaming tokens into the adapter."""
        num_workers = 50
        prompts = [
            "The quick brown fox",
            "jumps over the lazy dog.",
            "Hello world",
        ] * 17  # 51 prompts
        prompts = prompts[:num_workers]

        encoded_seqs = [self.adapter.encode(p) for p in prompts]

        def worker_task(worker_id: int) -> str:
            req_id = f"worker_{worker_id}"
            tokens = encoded_seqs[worker_id]
            streamed = []
            for tid in tokens:
                delta, _ = self.adapter.step_streaming(req_id, tid)
                if delta:
                    streamed.append(delta)
            final_flush = self.adapter.flush_streaming(req_id)
            if final_flush:
                streamed.append(final_flush)
            return "".join(streamed)

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(worker_task, range(num_workers)))

        for i in range(num_workers):
            expected = self.adapter.decode(encoded_seqs[i])
            self.assertEqual(results[i], expected)

        self.assertEqual(self.adapter.active_streaming_requests, 0)

    def test_async_vllm_streaming_worker(self) -> None:
        """Verify AsyncVLLMStreamingWorker consuming tokens asynchronously."""

        async def run_worker():
            worker = AsyncVLLMStreamingWorker(self.adapter, request_id="async_worker_1")
            tokens = self.adapter.encode("The quick brown fox")

            async def producer():
                for t in tokens:
                    await worker.put_token(t)
                    await asyncio.sleep(0.001)
                await worker.put_token(None)  # Sentinel for EOF

            async def consumer():
                chunks = []
                async for chunk in worker.stream_deltas():
                    chunks.append(chunk)
                return "".join(chunks)

            prod_task = asyncio.create_task(producer())
            text = await consumer()
            await prod_task
            return text

        result = asyncio.run(run_worker())
        self.assertEqual(result, "The quick brown fox")

    def test_abort_request_cleans_up(self) -> None:
        req_id = "abort_me"
        self.adapter.init_streaming_request(req_id)
        self.assertTrue(self.adapter.has_streaming_request(req_id))

        self.adapter.abort_request(req_id)
        self.assertFalse(self.adapter.has_streaming_request(req_id))
        self.assertIsNone(self.adapter.get_streaming_text(req_id))


if __name__ == "__main__":
    unittest.main()
