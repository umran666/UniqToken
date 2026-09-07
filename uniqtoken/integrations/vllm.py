"""
vLLM Custom Tokenizer & Streaming Detokenizer Integration for UniqToken.

Provides a high-throughput, zero-overhead adapter for vLLM inference backends:
- Non-blocking async batch encoding (`encode_batch_async`) and decoding (`decode_batch_async`).
- Thread-safe incremental streaming detokenization compatible with vLLM's `Detokenizer`.
- Byte-fallback UTF-8 buffering preventing replacement character (U+FFFD) corruptions during streaming.
- Concurrent async streaming worker supporting real-time token streams.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    overload,
)

from uniqtoken.streaming_decoder import StreamingDecoder
from uniqtoken.tokenizer import CustomTokenizer


class VLLMStreamingState:
    """
    Per-request thread-safe streaming detokenizer state for vLLM inference.

    Tracks accumulated generated token IDs, incremental decoded text, and maintains
    an incremental UTF-8 byte accumulator to prevent replacement character (U+FFFD)
    corruptions during token-by-token streaming. Uses prefix holdback for exact stop-string
    detection and truncation across multi-token boundaries.
    """

    def __init__(
        self,
        request_id: Union[str, int],
        decoder: StreamingDecoder,
        stop_strings: Optional[Sequence[str]] = None,
        include_stop_str_in_output: bool = False,
    ) -> None:
        self.request_id = request_id
        self.decoder = decoder
        self.stop_strings = [s for s in (stop_strings or []) if s]
        self.include_stop_str_in_output = include_stop_str_in_output
        self.generated_token_ids: List[int] = []
        self.decoded_text: str = ""
        self.emitted_len: int = 0
        self.is_finished: bool = False
        self.stop_reason: Optional[str] = None
        self._lock = threading.Lock()

    def _find_longest_stop_prefix_len(self, text: str) -> int:
        if not self.stop_strings:
            return 0
        max_prefix_len = 0
        for stop_str in self.stop_strings:
            max_k = min(len(text), len(stop_str) - 1)
            for k in range(max_k, 0, -1):
                if k <= max_prefix_len:
                    break
                if text.endswith(stop_str[:k]):
                    max_prefix_len = k
                    break
        return max_prefix_len

    def step(self, token_id: int) -> Tuple[str, bool]:
        """
        Feeds a single token ID into the streaming decoder.

        Returns:
            Tuple of (delta_text, is_finished).
            If a stop string was encountered, is_finished is True and delta_text is truncated accordingly.
        """
        with self._lock:
            if self.is_finished:
                return "", True

            self.generated_token_ids.append(token_id)
            delta = self.decoder.feed_token_id(token_id)
            if delta:
                self.decoded_text += delta

            # 1. Check for complete stop strings in decoded_text
            if self.stop_strings:
                earliest_match_pos = -1
                matched_stop_str = None
                for stop_str in self.stop_strings:
                    pos = self.decoded_text.find(stop_str)
                    if pos != -1 and (earliest_match_pos == -1 or pos < earliest_match_pos):
                        earliest_match_pos = pos
                        matched_stop_str = stop_str

                if earliest_match_pos != -1 and matched_stop_str is not None:
                    self.is_finished = True
                    self.stop_reason = matched_stop_str
                    stop_end = (
                        earliest_match_pos + len(matched_stop_str)
                        if self.include_stop_str_in_output
                        else earliest_match_pos
                    )
                    safe_len = max(self.emitted_len, stop_end)
                    emitted_delta = self.decoded_text[self.emitted_len : safe_len]
                    self.emitted_len = safe_len
                    self.decoded_text = self.decoded_text[:safe_len]
                    return emitted_delta, True

            # 2. Check for trailing partial prefixes to hold back
            holdback_len = self._find_longest_stop_prefix_len(self.decoded_text)
            safe_len = len(self.decoded_text) - holdback_len
            if safe_len > self.emitted_len:
                emitted_delta = self.decoded_text[self.emitted_len : safe_len]
                self.emitted_len = safe_len
                return emitted_delta, False

            return "", False

    def step_tokens(self, token_ids: Sequence[int]) -> Tuple[str, bool]:
        """Feeds multiple token IDs sequentially."""
        collected_deltas: List[str] = []
        for tid in token_ids:
            delta, is_finished = self.step(tid)
            if delta:
                collected_deltas.append(delta)
            if is_finished:
                return "".join(collected_deltas), True
        return "".join(collected_deltas), self.is_finished

    def flush(self) -> str:
        """Flushes any remaining buffered text/bytes and marks stream finished."""
        with self._lock:
            if self.is_finished:
                return ""
            delta = self.decoder.flush()
            if delta:
                self.decoded_text += delta

            # Check for complete stop strings on flush
            if self.stop_strings:
                earliest_match_pos = -1
                matched_stop_str = None
                for stop_str in self.stop_strings:
                    pos = self.decoded_text.find(stop_str)
                    if pos != -1 and (earliest_match_pos == -1 or pos < earliest_match_pos):
                        earliest_match_pos = pos
                        matched_stop_str = stop_str

                if earliest_match_pos != -1 and matched_stop_str is not None:
                    self.is_finished = True
                    self.stop_reason = matched_stop_str
                    stop_end = (
                        earliest_match_pos + len(matched_stop_str)
                        if self.include_stop_str_in_output
                        else earliest_match_pos
                    )
                    safe_len = max(self.emitted_len, stop_end)
                    emitted_delta = self.decoded_text[self.emitted_len : safe_len]
                    self.emitted_len = safe_len
                    self.decoded_text = self.decoded_text[:safe_len]
                    return emitted_delta

            # No stop string on flush: emit everything remaining
            emitted_delta = self.decoded_text[self.emitted_len :]
            self.emitted_len = len(self.decoded_text)
            self.is_finished = True
            return emitted_delta

    def get_text(self) -> str:
        """Returns the full decoded text produced so far."""
        with self._lock:
            return self.decoded_text[: self.emitted_len] if self.is_finished else self.decoded_text


class UniqTokenVLLMAdapter:
    """
    High-throughput vLLM tokenizer and detokenizer adapter for UniqToken.

    Provides:
    - Zero-overhead async batch encoding (`encode_batch_async`) and decoding (`decode_batch_async`).
    - Thread-safe streaming state management compatible with vLLM's `Detokenizer`.
    - Drop-in interface compatibility with Hugging Face PreTrainedTokenizer expectations in vLLM.
    """

    def __init__(
        self,
        tokenizer: CustomTokenizer,
        default_skip_special_tokens: bool = True,
    ) -> None:
        if not isinstance(tokenizer, CustomTokenizer):
            if not hasattr(tokenizer, "model") or not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
                raise TypeError(f"tokenizer must be a CustomTokenizer instance, got {type(tokenizer).__name__}")

        self.tokenizer = tokenizer
        self.default_skip_special_tokens = default_skip_special_tokens
        self.is_fast: bool = True

        # Extract special tokens and IDs
        self.pad_token: Optional[str] = self._find_special(["<|pad|>", "<pad>"])
        self.eos_token: Optional[str] = self._find_special(["<|eos|>", "</s>", "<|end_of_text|>"])
        self.bos_token: Optional[str] = self._find_special(["<|bos|>", "<s>", "<|begin_of_text|>"])
        self.unk_token: Optional[str] = getattr(self.tokenizer.model, "unk_token", None) or self._find_special(
            ["<|unk|>", "<unk>"]
        )

        token_to_id = self.tokenizer.model.token_to_id
        self.pad_token_id: Optional[int] = token_to_id.get(self.pad_token) if self.pad_token else None
        self.eos_token_id: Optional[int] = token_to_id.get(self.eos_token) if self.eos_token else None
        self.bos_token_id: Optional[int] = token_to_id.get(self.bos_token) if self.bos_token else None
        self.unk_token_id: Optional[int] = token_to_id.get(self.unk_token) if self.unk_token else None

        self.all_special_tokens: List[str] = list(getattr(self.tokenizer, "special_tokens", []) or [])
        self.all_special_ids: Set[int] = {token_to_id[tok] for tok in self.all_special_tokens if tok in token_to_id}

        # Thread-safe streaming state registry
        self._states_lock = threading.RLock()
        self._streaming_states: Dict[Union[str, int], VLLMStreamingState] = {}

    def _find_special(self, candidates: Sequence[str]) -> Optional[str]:
        special_set = set(getattr(self.tokenizer, "special_tokens", []) or [])
        token_to_id = getattr(self.tokenizer.model, "token_to_id", {})
        for c in candidates:
            if c in special_set or c in token_to_id:
                return c
        return None

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], **kwargs: Any) -> "UniqTokenVLLMAdapter":
        """Loads a CustomTokenizer from directory and returns a vLLM adapter."""
        tok = CustomTokenizer.load(path)
        return cls(tok, **kwargs)

    @classmethod
    def from_custom_tokenizer(cls, tokenizer: CustomTokenizer, **kwargs: Any) -> "UniqTokenVLLMAdapter":
        """Wraps an existing CustomTokenizer."""
        return cls(tokenizer, **kwargs)

    @property
    def vocab_size(self) -> int:
        """Returns vocabulary size."""
        return len(self.tokenizer.model.vocab)

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        """Returns token to ID vocabulary dictionary."""
        return dict(self.tokenizer.model.token_to_id)

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: Union[List[str], Tuple[str, ...]]) -> List[int]: ...

    def convert_tokens_to_ids(self, tokens: Union[str, Sequence[str]]) -> Union[int, List[int]]:
        """Converts a token or list of tokens to token IDs."""
        unk_id = self.unk_token_id if self.unk_token_id is not None else 0
        token_to_id = self.tokenizer.model.token_to_id
        if isinstance(tokens, str):
            return token_to_id.get(tokens, unk_id)
        return [token_to_id.get(t, unk_id) for t in tokens]

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str: ...

    @overload
    def convert_ids_to_tokens(self, ids: Union[List[int], Tuple[int, ...]]) -> List[str]: ...

    def convert_ids_to_tokens(self, ids: Union[int, Sequence[int]]) -> Union[str, List[str]]:
        """Converts a token ID or list of IDs to token strings."""
        unk_tok = self.unk_token or "<|unk|>"
        id_to_token = self.tokenizer.model.id_to_token
        if isinstance(ids, int):
            return id_to_token.get(ids, unk_tok)
        return [id_to_token.get(i, unk_tok) for i in ids]

    # -------------------------------------------------------------------------
    # Encoding & Batch Encoding (Sync & Async)
    # -------------------------------------------------------------------------

    def encode(
        self,
        prompt: str,
        add_special_tokens: bool = False,
        allowed_special: Union[str, Set[str], List[str]] = "all",
    ) -> List[int]:
        """Encodes a single prompt string to a list of token IDs."""
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        ids = self.tokenizer.encode_to_ids(prompt, allowed_special=allowed_special)
        if add_special_tokens:
            prefix = [self.bos_token_id] if self.bos_token_id is not None else []
            suffix = [self.eos_token_id] if self.eos_token_id is not None else []
            return prefix + ids + suffix
        return ids

    def encode_batch(
        self,
        prompts: Sequence[str],
        add_special_tokens: bool = False,
        num_workers: Optional[int] = None,
        allowed_special: Union[str, Set[str], List[str]] = "all",
    ) -> List[List[int]]:
        """Batch encodes multiple prompts to lists of token IDs."""
        if not prompts:
            return []
        batch_ids = self.tokenizer.encode_to_ids_batch(
            prompts,
            allowed_special=allowed_special,
            num_workers=num_workers,
        )
        if add_special_tokens and (self.bos_token_id is not None or self.eos_token_id is not None):
            prefix = [self.bos_token_id] if self.bos_token_id is not None else []
            suffix = [self.eos_token_id] if self.eos_token_id is not None else []
            return [prefix + seq + suffix for seq in batch_ids]
        return batch_ids

    async def encode_batch_async(
        self,
        prompts: Sequence[str],
        add_special_tokens: bool = False,
        num_workers: Optional[int] = None,
        executor: Optional[concurrent.futures.Executor] = None,
        allowed_special: Union[str, Set[str], List[str]] = "all",
    ) -> List[List[int]]:
        """
        Asynchronously batch encodes prompts without blocking the event loop.
        Dispatches CPU-bound tokenization to a thread executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.encode_batch,
            prompts,
            add_special_tokens,
            num_workers,
            allowed_special,
        )

    # -------------------------------------------------------------------------
    # Decoding & Batch Decoding (Sync & Async)
    # -------------------------------------------------------------------------

    def decode(
        self,
        tokens: Sequence[int],
        skip_special_tokens: Optional[bool] = None,
    ) -> str:
        """Decodes token IDs into a text string."""
        if not isinstance(tokens, (list, tuple)):
            tokens = list(tokens)
        skip = self.default_skip_special_tokens if skip_special_tokens is None else skip_special_tokens
        if skip and self.all_special_ids:
            filtered = [t for t in tokens if t not in self.all_special_ids]
        else:
            filtered = list(tokens)
        return self.tokenizer.decode(filtered)

    def decode_batch(
        self,
        sequences: Sequence[Sequence[int]],
        skip_special_tokens: Optional[bool] = None,
        num_workers: Optional[int] = None,
    ) -> List[str]:
        """Batch decodes token ID sequences into strings."""
        if not sequences:
            return []
        skip = self.default_skip_special_tokens if skip_special_tokens is None else skip_special_tokens
        if skip and self.all_special_ids:
            filtered_sequences = [[t for t in seq if t not in self.all_special_ids] for seq in sequences]
        else:
            filtered_sequences = [list(seq) for seq in sequences]
        return self.tokenizer.decode_batch(filtered_sequences, num_workers=num_workers)

    async def decode_batch_async(
        self,
        sequences: Sequence[Sequence[int]],
        skip_special_tokens: Optional[bool] = None,
        num_workers: Optional[int] = None,
        executor: Optional[concurrent.futures.Executor] = None,
    ) -> List[str]:
        """
        Asynchronously batch decodes sequences without blocking the event loop.
        Dispatches decoding to a thread executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            self.decode_batch,
            sequences,
            skip_special_tokens,
            num_workers,
        )

    # -------------------------------------------------------------------------
    # Thread-Safe Streaming Detokenization (vLLM Detokenizer protocol)
    # -------------------------------------------------------------------------

    def create_streaming_decoder(self, skip_special_tokens: Optional[bool] = None) -> StreamingDecoder:
        """Creates an incremental StreamingDecoder instance backed by the underlying tokenizer."""
        skip = self.default_skip_special_tokens if skip_special_tokens is None else skip_special_tokens
        return self.tokenizer.get_streaming_decoder(skip_special_tokens=skip)

    def init_streaming_request(
        self,
        request_id: Union[str, int],
        stop_strings: Optional[Sequence[str]] = None,
        include_stop_str_in_output: bool = False,
        skip_special_tokens: Optional[bool] = None,
    ) -> VLLMStreamingState:
        """Registers a new active streaming request."""
        with self._states_lock:
            decoder = self.create_streaming_decoder(skip_special_tokens=skip_special_tokens)
            state = VLLMStreamingState(
                request_id=request_id,
                decoder=decoder,
                stop_strings=stop_strings,
                include_stop_str_in_output=include_stop_str_in_output,
            )
            self._streaming_states[request_id] = state
            return state

    def step_streaming(
        self,
        request_id: Union[str, int],
        token_id: int,
        stop_strings: Optional[Sequence[str]] = None,
        skip_special_tokens: Optional[bool] = None,
    ) -> Tuple[str, bool]:
        """
        Steps a single token for the given request ID in a thread-safe manner.

        Returns:
            Tuple of (delta_text, is_finished).
        """
        with self._states_lock:
            state = self._streaming_states.get(request_id)
            if state is None:
                state = self.init_streaming_request(
                    request_id,
                    stop_strings=stop_strings,
                    skip_special_tokens=skip_special_tokens,
                )
        return state.step(token_id)

    def step_streaming_batch(
        self,
        items: Sequence[Tuple[Union[str, int], int]],
    ) -> List[Tuple[Union[str, int], str, bool]]:
        """
        Steps multiple token emissions across different requests.

        Returns:
            List of (request_id, delta_text, is_finished).
        """
        results: List[Tuple[Union[str, int], str, bool]] = []
        for req_id, token_id in items:
            delta, is_finished = self.step_streaming(req_id, token_id)
            results.append((req_id, delta, is_finished))
        return results

    def flush_streaming(
        self,
        request_id: Union[str, int],
        cleanup: bool = True,
    ) -> str:
        """Flushes trailing bytes/characters and optionally cleans up request state."""
        with self._states_lock:
            state = self._streaming_states.get(request_id)
            if state is None:
                return ""
            if cleanup:
                del self._streaming_states[request_id]
        return state.flush()

    def finish_request(self, request_id: Union[str, int]) -> str:
        """Finalizes a streaming request, flushes remaining text, and frees state."""
        return self.flush_streaming(request_id, cleanup=True)

    def abort_request(self, request_id: Union[str, int]) -> None:
        """Aborts and removes streaming state for a request without flushing."""
        with self._states_lock:
            self._streaming_states.pop(request_id, None)

    def has_streaming_request(self, request_id: Union[str, int]) -> bool:
        """Checks if a request is actively tracked."""
        with self._states_lock:
            return request_id in self._streaming_states

    def get_streaming_text(self, request_id: Union[str, int]) -> Optional[str]:
        """Gets the accumulated text for a request so far."""
        with self._states_lock:
            state = self._streaming_states.get(request_id)
            return state.get_text() if state is not None else None

    @property
    def active_streaming_requests(self) -> int:
        """Number of active streaming requests currently tracked."""
        with self._states_lock:
            return len(self._streaming_states)

    def clear_all_streaming(self) -> None:
        """Clears all active streaming states."""
        with self._states_lock:
            self._streaming_states.clear()

    # -------------------------------------------------------------------------
    # vLLM Detokenizer Incremental Protocol
    # -------------------------------------------------------------------------

    def detokenize_incrementally(
        self,
        request_id: Union[str, int],
        new_token_id: int,
    ) -> str:
        """
        vLLM incremental detokenizer interface.
        Feeds new_token_id and returns the new string delta.
        """
        delta, _ = self.step_streaming(request_id, new_token_id)
        return delta


class VLLMDetokenizer:
    """
    Mock/drop-in vLLM Detokenizer simulation.

    Mimics the exact consumption pattern of vLLM's internal Detokenizer worker:
    maintains per-request decode streams, applies incremental detokenization,
    tracks stop words, and flushes on request completion.
    """

    def __init__(
        self,
        adapter: UniqTokenVLLMAdapter,
        default_stop_strings: Optional[Sequence[str]] = None,
    ) -> None:
        self.adapter = adapter
        self.default_stop_strings = list(default_stop_strings or [])

    def init_request(
        self,
        request_id: Union[str, int],
        stop_strings: Optional[Sequence[str]] = None,
        include_stop_str_in_output: bool = False,
    ) -> None:
        """Registers a new generation request."""
        stops = stop_strings if stop_strings is not None else self.default_stop_strings
        self.adapter.init_streaming_request(
            request_id=request_id,
            stop_strings=stops,
            include_stop_str_in_output=include_stop_str_in_output,
        )

    def step(self, request_id: Union[str, int], new_token_id: int) -> Tuple[str, bool]:
        """
        Consumes one token emitted by the model for request_id.

        Returns:
            (delta_text, is_finished)
        """
        return self.adapter.step_streaming(request_id, new_token_id)

    def flush(self, request_id: Union[str, int]) -> str:
        """Flushes remaining text for request_id and cleans up."""
        return self.adapter.flush_streaming(request_id, cleanup=True)

    def abort(self, request_id: Union[str, int]) -> None:
        """Aborts the request."""
        self.adapter.abort_request(request_id)

    def get_output_text(self, request_id: Union[str, int]) -> Optional[str]:
        """Gets full accumulated output text."""
        return self.adapter.get_streaming_text(request_id)

    @property
    def active_requests(self) -> int:
        """Number of currently tracked requests."""
        return self.adapter.active_streaming_requests


class AsyncVLLMStreamingWorker:
    """
    Asynchronous Streaming Worker for vLLM Inference.

    Consumes token streams from an asyncio Queue and yields decoded text deltas
    without blocking the event loop.
    """

    def __init__(
        self,
        adapter: UniqTokenVLLMAdapter,
        request_id: Union[str, int],
        queue: Optional[asyncio.Queue[Optional[int]]] = None,
        stop_strings: Optional[Sequence[str]] = None,
    ) -> None:
        self.adapter = adapter
        self.request_id = request_id
        self.queue: asyncio.Queue[Optional[int]] = queue if queue is not None else asyncio.Queue()
        self.adapter.init_streaming_request(request_id, stop_strings=stop_strings)

    async def put_token(self, token_id: Optional[int]) -> None:
        """Pushes a token ID into the worker's stream. Send None to terminate."""
        await self.queue.put(token_id)

    async def stream_deltas(self) -> AsyncIterator[str]:
        """
        Asynchronously yields text deltas as tokens arrive.
        Terminates on None or when a stop condition is triggered.
        """
        try:
            while True:
                token_id = await self.queue.get()
                if token_id is None:
                    # End of stream
                    final_delta = self.adapter.flush_streaming(self.request_id, cleanup=True)
                    if final_delta:
                        yield final_delta
                    break

                delta, is_finished = self.adapter.step_streaming(self.request_id, token_id)
                if delta:
                    yield delta
                if is_finished:
                    self.adapter.flush_streaming(self.request_id, cleanup=True)
                    break
        finally:
            self.adapter.abort_request(self.request_id)
