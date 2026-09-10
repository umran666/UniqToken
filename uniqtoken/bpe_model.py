from __future__ import annotations

import heapq
import random
from typing import Dict, List, Optional, Set, Tuple

from .byte_codec import ByteFallbackEngine, validate_dropout_prob


class BPEModel:
    """
    Byte-Pair Encoding (BPE) Subword Model.

    Implements iterative pair merging based on learned merge priority ranks
    (Tiktoken / GPT-4 / BPE standard). The inference path uses a
    rank-priority heap over adjacent pairs, giving O(len * log len) per
    word instead of the naive O(merges * len).
    """

    def __init__(
        self,
        vocab: Set[str],
        token_to_id: Dict[str, int],
        id_to_token: Dict[int, str],
        merges: Dict[Tuple[str, str], int],
        special_tokens: Optional[List[str]] = None,
        byte_fallback: bool = True,
    ):
        self.vocab = vocab
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.merges = merges
        self.special_tokens = list(special_tokens or [])
        self.byte_fallback = byte_fallback
        self._validate_byte_fallback()
        if "<|unk|>" in token_to_id:
            self._unk_token: Optional[str] = "<|unk|>"
        elif self.special_tokens:
            self._unk_token = self.special_tokens[0]
        else:
            self._unk_token = None
        if " " in token_to_id and " " in vocab:
            self._space_token = " "
        elif byte_fallback:
            self._space_token = ByteFallbackEngine.byte_to_token(32)
        else:
            self._space_token = " "

    def _validate_byte_fallback(self) -> None:
        if self.byte_fallback:
            required = {ByteFallbackEngine.byte_to_token(b) for b in range(256)}
            if not required.issubset(self.vocab) or not required.issubset(self.token_to_id):
                raise ValueError("byte_fallback=True requires all 256 byte tokens with integer IDs")
            if any(self.id_to_token.get(self.token_to_id[t]) != t for t in required):
                raise ValueError("byte fallback IDs must map back to their byte tokens")

    @property
    def vocab_size(self) -> int:
        # Imported rank-based vocabularies can be sparse, so size is the ID
        # span rather than the number of distinct token strings.
        return max(self.token_to_id.values(), default=-1) + 1

    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        return set(zip(word[:-1], word[1:]))

    def _build_symbols(self, word: str) -> List[str]:
        if not word:
            return []
        self._validate_byte_fallback()
        symbols: List[str] = []
        for char in word:
            if char in self.vocab:
                symbols.append(char)
            elif self.byte_fallback:
                symbols.extend(ByteFallbackEngine.char_to_byte_tokens(char))
            elif self._unk_token is not None:
                symbols.append(self._unk_token)
            else:
                symbols.append(char)
        return symbols

    def _encode_word_heap(self, symbols: List[str], dropout_prob: float = 0.0) -> List[str]:
        """Rank-priority BPE encode using a min-heap of adjacent pairs.

        Each heap entry is ``(rank, counter, left, right)``. Popping the
        smallest-rank pair applies a merge, and the two neighbouring
        pairs (left-of-merged and merged-of-right) are pushed with their
        cached ranks. The ``counter`` makes heap entries unique without
        hashing the strings on every comparison. Positions whose merge
        already happened are removed lazily: a stale entry is detected
        by comparing the popped rank against the live ``self.merges`` value
        for the popped pair, which is a fast dict lookup.

        With ``dropout_prob > 0`` each live merge candidate is skipped
        with that probability (BPE dropout, Provilkov et al. 2020) without
        mutating the symbol list, so the constituent symbols stay intact.
        A dropped candidate is discarded, making the drop final for this
        call: a later merge of a neighbouring pair never resurrects it.
        """
        if len(symbols) <= 1:
            return list(symbols)

        ranks = self.merges
        if not ranks:
            return list(symbols)

        # Keep a doubly-linked list of live symbol positions. A merge updates
        # only its two neighboring pairs, so each heap pop is O(log n) instead
        # of rebuilding and rescanning the entire symbol list.
        syms: List[str] = list(symbols)
        prev = [i - 1 for i in range(len(syms))]
        next_pos = [i + 1 if i + 1 < len(syms) else -1 for i in range(len(syms))]
        alive = [True] * len(syms)
        heap: List[Tuple[int, int, int, int]] = []
        counter = 0
        for i in range(len(syms) - 1):
            pair = (syms[i], syms[i + 1])
            rank = ranks.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, counter, i, i + 1))
                counter += 1

        while heap:
            rank, _, left_idx, right_idx = heapq.heappop(heap)
            if not alive[left_idx] or not alive[right_idx] or next_pos[left_idx] != right_idx:
                continue
            live_rank = ranks.get((syms[left_idx], syms[right_idx]))
            if live_rank != rank:
                continue

            if dropout_prob > 0.0 and random.random() < dropout_prob:
                continue  # BPE dropout: keep both constituent symbols intact

            syms[left_idx] += syms[right_idx]
            alive[right_idx] = False
            right_next = next_pos[right_idx]
            next_pos[left_idx] = right_next
            if right_next != -1:
                prev[right_next] = left_idx

            left_prev = prev[left_idx]
            if left_prev != -1:
                pair_rank = ranks.get((syms[left_prev], syms[left_idx]))
                if pair_rank is not None:
                    heapq.heappush(heap, (pair_rank, counter, left_prev, left_idx))
                    counter += 1
            if right_next != -1:
                pair_rank = ranks.get((syms[left_idx], syms[right_next]))
                if pair_rank is not None:
                    heapq.heappush(heap, (pair_rank, counter, left_idx, right_next))
                    counter += 1

        result: List[str] = []
        index = next((i for i, is_alive in enumerate(alive) if is_alive and prev[i] == -1), -1)
        while index != -1:
            result.append(syms[index])
            index = next_pos[index]
        return result

    def _encode_word(self, word: str, dropout_prob: float = 0.0) -> List[str]:
        symbols = self._build_symbols(word)
        if len(symbols) <= 1:
            return symbols
        return self._encode_word_heap(symbols, dropout_prob=dropout_prob)

    def encode(self, text: str, dropout_prob: float = 0.0) -> List[str]:
        """
        Segments text by applying BPE merges on whitespace-delimited word tokens.

        With ``dropout_prob > 0`` each eligible merge candidate is
        independently skipped with that probability (BPE dropout,
        Provilkov et al. 2020); a dropped merge is final for the call.
        ``0.0`` (default) keeps tokenization fully deterministic.
        """
        validate_dropout_prob(dropout_prob)
        if not text:
            return []

        words = text.split(" ")
        tokens: List[str] = []
        for idx, word in enumerate(words):
            if idx > 0:
                tokens.append(self._space_token)
            tokens.extend(self._encode_word(word, dropout_prob=dropout_prob))
        return tokens

    def encode_to_ids(self, text: str, dropout_prob: float = 0.0) -> List[int]:
        """Encodes text to token IDs; ``dropout_prob`` behaves as in :meth:`encode`."""
        tokens = self.encode(text, dropout_prob=dropout_prob)
        return [self.token_to_id[t] for t in tokens]

    def decode(self, token_ids: List[int], space_char: str = "\u2581", strict: bool = False) -> str:
        """
        Decodes integer token IDs back to a human-readable string.

        strict=False (default, lenient): unknown IDs are skipped silently.
        strict=True: raises ValueError naming the first invalid ID.
        """
        tokens: List[str] = []
        for t in token_ids:
            tok = self.id_to_token.get(t)
            if tok is None:
                if strict:
                    raise ValueError(f"token id {t} is not in the model vocabulary")
                continue  # lenient: unknown IDs contribute nothing
            tokens.append(tok)
        return ByteFallbackEngine.decode_tokens(tokens, space_char=space_char)
