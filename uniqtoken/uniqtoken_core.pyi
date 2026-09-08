from typing import Dict, List, Optional, Sequence, Set, Tuple

class RustPrefixTrie:
    def __init__(self) -> None: ...
    def insert(self, token: str, log_p: float, token_id: Optional[int] = None) -> None: ...
    def common_prefix_search_chars(
        self, chars: Sequence[str], start_idx: int = 0
    ) -> List[Tuple[str, Optional[int], float, int]]: ...
    def clear_seg_cache(self) -> None: ...
    def seg_cache_len(self) -> int: ...

class RustTokenizer:
    def __init__(
        self,
        vocab: Optional[List[Tuple[str, float, int]]] = None,
        space_char: str = "\u2581",
        byte_fallback: bool = True,
    ) -> None: ...
    def from_vocab(self, vocab: List[Tuple[str, float, int]]) -> None: ...
    def encode(self, text: str) -> List[str]: ...
    def encode_batch(self, texts: Sequence[str]) -> List[List[str]]: ...
    def encode_ids(self, text: str) -> List[int]: ...
    def encode_ids_batch(self, texts: Sequence[str]) -> List[List[int]]: ...

class ViterbiSpan:
    token: str
    token_id: Optional[int]
    start: int
    end: int

def rust_viterbi_decode(
    text: str,
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    max_edges_per_node: Optional[int] = None,
) -> List[ViterbiSpan]: ...
def rust_viterbi_decode_batch(
    texts: Sequence[str],
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    max_edges_per_node: Optional[int] = None,
) -> List[List[ViterbiSpan]]: ...
def rust_encode_tokens_batch(
    texts: Sequence[str],
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    max_edges_per_node: Optional[int] = None,
) -> List[List[str]]: ...
def rust_encode_ids_batch(
    texts: Sequence[str],
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    max_edges_per_node: Optional[int] = None,
) -> List[List[int]]: ...
def rust_encode_text_batch(
    texts: Sequence[str],
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    space_char: str = "\u2581",
) -> List[List[int]]: ...
def rust_encode_text_native(
    text: str,
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    space_char: str = "\u2581",
    normalize_unicode: bool = True,
    normalize_unicode_spaces: bool = True,
    normalize_punctuation: bool = False,
    lowercase: bool = False,
    collapse_whitespaces: bool = False,
    strip_whitespace: bool = False,
) -> List[str]: ...
def rust_encode_text_native_batch(
    texts: Sequence[str],
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    space_char: str = "\u2581",
    normalize_unicode: bool = True,
    normalize_unicode_spaces: bool = True,
    normalize_punctuation: bool = False,
    lowercase: bool = False,
    collapse_whitespaces: bool = False,
    strip_whitespace: bool = False,
) -> List[List[str]]: ...
def rust_encode_text_native_ids(
    text: str,
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    space_char: str = "\u2581",
    normalize_unicode: bool = True,
    normalize_unicode_spaces: bool = True,
    normalize_punctuation: bool = False,
    lowercase: bool = False,
    collapse_whitespaces: bool = False,
    strip_whitespace: bool = False,
) -> List[int]: ...
def rust_encode_text_native_ids_batch(
    texts: Sequence[str],
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
    space_char: str = "\u2581",
    normalize_unicode: bool = True,
    normalize_unicode_spaces: bool = True,
    normalize_punctuation: bool = False,
    lowercase: bool = False,
    collapse_whitespaces: bool = False,
    strip_whitespace: bool = False,
) -> List[List[int]]: ...
def rust_diagnostic_batch(
    texts: Sequence[str],
    trie: RustPrefixTrie,
    byte_fallback: bool = True,
) -> Dict[str, float]: ...
def rust_forward_backward_expectations(
    text: str,
    trie: RustPrefixTrie,
    freq: float = 1.0,
) -> Tuple[Dict[str, float], float]: ...
def rust_normalize(
    text: str,
    space_char: str = "\u2581",
    normalize_unicode: bool = True,
    normalize_unicode_spaces: bool = True,
    normalize_punctuation: bool = False,
    lowercase: bool = False,
    collapse_whitespaces: bool = False,
    strip_whitespace: bool = False,
) -> str: ...
def rust_normalize_with_alignment(
    text: str,
    space_char: str = "\u2581",
    normalize_unicode: bool = True,
    normalize_unicode_spaces: bool = True,
    normalize_punctuation: bool = False,
    lowercase: bool = False,
    collapse_whitespaces: bool = False,
    strip_whitespace: bool = False,
) -> Tuple[str, List[Tuple[int, int]]]: ...
def rust_mine_ngrams(
    chunk_counts: Dict[str, int],
    max_ngram_length: int,
    special_tokens: Optional[Set[str]] = None,
) -> Dict[str, int]: ...
def rust_pre_tokenize(
    text: str,
) -> List[str]: ...
