from __future__ import annotations

import functools
import re
import unicodedata
import difflib
from dataclasses import dataclass
from typing import Iterator, List, Literal, Optional, Sequence, Tuple, Union

DigitChunking = Literal["block3", "single", "greedy"]

try:
    import uniqtoken_core as _caliper_core

    _HAS_RUST_NORM = hasattr(_caliper_core, "rust_normalize_with_alignment")
except ImportError:
    try:
        import caliper_core as _caliper_core  # type: ignore[no-redef]

        _HAS_RUST_NORM = hasattr(_caliper_core, "rust_normalize_with_alignment")
    except ImportError:
        _caliper_core = None  # type: ignore[assignment]
        _HAS_RUST_NORM = False


@dataclass(frozen=True)
class PreToken:
    """An atomic normalized chunk with normalized and original-text spans."""

    text: str
    start: int
    end: int
    raw_span: Tuple[int, int]

    @property
    def span(self) -> Tuple[int, int]:
        return (self.start, self.end)

    @property
    def norm_span(self) -> Tuple[int, int]:
        return self.span


class Normalizer:
    """
    Standardizes raw text before tokenization.

    NOTE ON REVERSIBILITY:
    - Normalization with NFKC is *canonical*, not byte-exact lossless.
      Compatibility characters (e.g. 'ﬁ' -> 'fi', '²' -> '2') are intentionally transformed.
    - If exact raw string reconstruction is required, disable NFKC (`normalize_unicode=False`).
    """

    PUNCT_MAP = str.maketrans(
        {
            "“": '"',
            "”": '"',
            "„": '"',
            "‘": "'",
            "’": "'",
            "‚": "'",
            "—": "-",
            "–": "-",
            "−": "-",
            "…": "...",
        }
    )

    UNICODE_SPACES = re.compile(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]")
    # ponytail: set lookup O(1) vs regex fullmatch per char; upgrade to table if more spaces
    _UNICODE_SPACE_SET = frozenset(
        [chr(0x00A0), chr(0x1680)] + [chr(cp) for cp in range(0x2000, 0x200B)] + [chr(0x202F), chr(0x205F), chr(0x3000)]
    )
    _ESCAPE_PREFIX = "\ue000"
    _ESCAPED_METASPACE = "\ue001"

    def __init__(
        self,
        space_char: str = "\u2581",
        lowercase: bool = False,
        normalize_unicode: bool = True,
        normalize_punctuation: bool = False,
        normalize_unicode_spaces: bool = True,
        collapse_whitespaces: bool = False,
        strip_whitespace: bool = False,
        casefold: bool = False,
    ):
        if not isinstance(space_char, str) or len(space_char) != 1:
            raise ValueError("space_char must be exactly one character")
        if space_char in {self._ESCAPE_PREFIX, self._ESCAPED_METASPACE}:
            raise ValueError("space_char conflicts with reserved metaspace escape characters")
        self.space_char = space_char
        self.lowercase = lowercase
        self.casefold = casefold
        self.normalize_unicode = normalize_unicode
        self.normalize_punctuation = normalize_punctuation
        self.normalize_unicode_spaces = normalize_unicode_spaces
        self.collapse_whitespaces = collapse_whitespaces
        self.strip_whitespace = strip_whitespace

    @staticmethod
    def _expand(value: str, span: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        return [(char, span) for char in value]

    @staticmethod
    def _nfkc_units(text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """Normalize the complete string and conservatively retain source spans."""
        normalized = unicodedata.normalize("NFKC", text)
        if normalized == text:
            return [(char, (i, i + 1)) for i, char in enumerate(text)]

        chars = list(text)
        n = len(chars)
        prefix_lengths = [0]
        for i in range(n):
            prefix = "".join(chars[: i + 1])
            prefix_lengths.append(len(unicodedata.normalize("NFKC", prefix)))
        normalized_chars = list(normalized)
        spans = [(0, 0)] * len(normalized_chars)
        for i in range(n):
            start = prefix_lengths[i]
            end = prefix_lengths[i + 1]
            if end > start:
                for idx in range(start, end):
                    spans[idx] = (i, i + 1)
            elif start > 0:
                for idx in range(start):
                    if spans[idx][1] == i:
                        spans[idx] = (spans[idx][0], i + 1)
        for idx in range(len(spans)):
            if spans[idx][0] == spans[idx][1]:
                spans[idx] = (0, 0) if n == 0 else (0, n)
        return list(zip(normalized_chars, spans))

    def normalize_with_alignment(self, text: str) -> Tuple[str, List[Tuple[int, int]]]:
        """Normalizes text and maps every output character to its raw source span."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        # ponytail: Rust normalizer with exact parity; Python fallback if mismatch
        if _HAS_RUST_NORM and not self.casefold:
            assert _caliper_core is not None
            try:
                res = _caliper_core.rust_normalize_with_alignment(
                    text,
                    self.space_char,
                    self.normalize_unicode,
                    self.normalize_unicode_spaces,
                    self.normalize_punctuation,
                    self.lowercase,
                    self.collapse_whitespaces,
                    self.strip_whitespace,
                )
                # ponytail: no per-element tuple() — Rust already returns List[Tuple]
                return res[0], res[1]
            except (ValueError, AttributeError, ImportError, TypeError):
                pass

        if self.normalize_unicode:
            units = self._nfkc_units(text)
        else:
            units = [(char, (i, i + 1)) for i, char in enumerate(text)]

        if self.normalize_unicode_spaces:
            # set lookup vs regex fullmatch per char
            _space_set = self._UNICODE_SPACE_SET
            units = [(" " if char in _space_set else char, span) for char, span in units]

        if self.normalize_punctuation:
            _map = self.PUNCT_MAP
            translated: List[Tuple[str, Tuple[int, int]]] = []
            for char, span in units:
                t = char.translate(_map)
                if len(t) == 1:
                    translated.append((t, span))
                elif t:
                    for c in t:
                        translated.append((c, span))
            units = translated

        if self.lowercase or self.casefold:
            lowered: List[Tuple[str, Tuple[int, int]]] = []
            for char, span in units:
                lo = char.casefold() if self.casefold else char.lower()
                if len(lo) == 1:
                    lowered.append((lo, span))
                elif lo:
                    for c in lo:
                        lowered.append((c, span))
            units = lowered

        if self.collapse_whitespaces:
            collapsed: List[Tuple[str, Tuple[int, int]]] = []
            i = 0
            ulen = len(units)
            while i < ulen:
                char, span = units[i]
                if char not in {" ", "\t"}:
                    collapsed.append((char, span))
                    i += 1
                    continue
                end = i + 1
                while end < ulen and units[end][0] in {" ", "\t"}:
                    end += 1
                collapsed.append((" ", (span[0], units[end - 1][1][1])))
                i = end
            units = collapsed

        if self.strip_whitespace:
            start = 0
            end = len(units)
            while start < end and units[start][0].isspace():
                start += 1
            while end > start and units[end - 1][0].isspace():
                end -= 1
            units = units[start:end]

        # metaspace escape — single pass, avoid _expand
        escaped: List[Tuple[str, Tuple[int, int]]] = []
        sc = self.space_char
        ep = self._ESCAPE_PREFIX
        em = self._ESCAPED_METASPACE
        ep2 = ep * 2
        esc_seq = ep + em
        for char, span in units:
            if char == ep:
                # two chars share same span
                escaped.append((ep, span))
                escaped.append((ep, span))
            elif char == sc:
                escaped.append((ep, span))
                escaped.append((em, span))
            elif char == " ":
                escaped.append((sc, span))
            else:
                escaped.append((char, span))

        # one final join
        return "".join(c for c, _ in escaped), [s for _, s in escaped]

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if _HAS_RUST_NORM and not self.casefold:
            assert _caliper_core is not None
            try:
                return _caliper_core.rust_normalize(
                    text,
                    self.space_char,
                    self.normalize_unicode,
                    self.normalize_unicode_spaces,
                    self.normalize_punctuation,
                    self.lowercase,
                    self.collapse_whitespaces,
                    self.strip_whitespace,
                )
            except (ValueError, AttributeError, ImportError, TypeError):
                pass
        return self.normalize_with_alignment(text)[0]

    def restore_escaped_metaspace(self, text: str) -> str:
        """Restores literal metaspace and escape-prefix characters after decoding."""
        restored: List[str] = []
        i = 0
        while i < len(text):
            if text[i] != self._ESCAPE_PREFIX or i + 1 >= len(text):
                restored.append(text[i])
                i += 1
                continue
            marker = text[i + 1]
            if marker == self._ESCAPED_METASPACE:
                restored.append(self.space_char)
                i += 2
            elif marker == self._ESCAPE_PREFIX:
                restored.append(self._ESCAPE_PREFIX)
                i += 2
            else:
                restored.append(text[i])
                i += 1
        return "".join(restored)


@functools.lru_cache(maxsize=128)
def _get_cached_regex(pattern: str) -> re.Pattern[str]:
    """Compiles and memoizes regular expression patterns across pre-tokenizer instances."""
    return re.compile(pattern)


try:
    import regex as _regex

    # Extended grapheme clusters (UAX #29) with current-Unicode data, mirroring
    # the Rust engine's unicode-segmentation crate.
    _X_RE = _regex.compile(r"\X")
    _MARK_RE = _regex.compile(r"\p{M}")
except ImportError:  # pragma: no cover
    # ponytail: without the `regex` package the fallbacks below use
    # interpreter-Unicode `unicodedata`, which can misclassify marks added in
    # newer Unicode versions (e.g. U+0897 on older CPythons) and drift from
    # the Rust engine. `regex` is the parity contract / upgrade path.
    _X_RE = None
    _MARK_RE = None


def _grapheme_boundaries(text: str) -> set:
    """Offsets of extended-grapheme-cluster boundaries in `text`."""
    if _X_RE is not None:
        return {m.start() for m in _X_RE.finditer(text)} | {len(text)}
    # Fallback without `regex`: every codepoint boundary (no cluster snapping).
    return set(range(len(text) + 1))


def _is_mark(ch: str) -> bool:
    r"""True for Unicode combining marks (\p{M}); current-Unicode aware."""
    if _MARK_RE is not None:
        return _MARK_RE.match(ch) is not None
    return unicodedata.category(ch).startswith("M")


def _snap_spans_to_graphemes(text: str, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Port of `pipeline.rs::snap_spans_to_graphemes` (UAX #29 snapping).

    Offsets are Python string indices (chars); the Rust original uses bytes,
    which is equivalent since every boundary, span, and advance here shares
    the same unit. Mirroring the reference keeps Python/rust pre-tokenization
    identical even on pathological inputs.
    """
    if not spans:
        return []
    boundaries = _grapheme_boundaries(text)
    out: List[Tuple[int, int]] = []
    i = 0
    n = len(spans)
    while i < n:
        s, e = spans[i]
        if s not in boundaries:
            if out:
                last_s, last_e = out[-1]
                if e > last_e:
                    last_e = e
                while last_e not in boundaries and last_e < len(text):
                    last_e += 1
                while i + 1 < n and spans[i + 1][0] < last_e:
                    i += 1
                    if spans[i][1] > last_e:
                        last_e = spans[i][1]
                        while last_e not in boundaries and last_e < len(text):
                            last_e += 1
                out[-1] = (last_s, last_e)
                i += 1
                continue
        while e not in boundaries and e < len(text):
            e += 1
        while i + 1 < n and spans[i + 1][0] < e:
            i += 1
            if spans[i][1] > e:
                e = spans[i][1]
                while e not in boundaries and e < len(text):
                    e += 1
        # Degenerate leading orphan (text starts with \p{M}): fuse forward,
        # absorbing every span under the growing cluster end (matches Rust).
        if not out and s < e:
            if _is_mark(text[s]) and i + 1 < n:
                e = spans[i + 1][1]
                while e not in boundaries and e < len(text):
                    e += 1
                j = i + 1
                while j + 1 < n and spans[j + 1][0] < e:
                    j += 1
                    if spans[j][1] > e:
                        e = spans[j][1]
                        while e not in boundaries and e < len(text):
                            e += 1
                i = j + 1
                out.append((s, e))
                continue
        out.append((s, e))
        i += 1
    return out


class RegexPreTokenizer:
    """
    Offset-preserving, regex-based Pre-Tokenizer.

    Uses compiled C-level regex iteration (`finditer`) to slice input text into
    atomic chunks while preserving character spans for downstream tasks.
    """

    def __init__(
        self,
        space_char: str = "\u2581",
        split_digits: bool = False,
        split_punctuation: bool = True,
        keep_special_tokens: bool = True,
        special_token_pattern: str = r"<\|[^\s|]+\|>",
        hex_literals: bool = True,
        digit_chunk_size: Optional[int] = None,
        digit_chunking: DigitChunking = "block3",
        preset: Optional[str] = None,
    ):
        if digit_chunking not in ("block3", "single", "greedy"):
            raise ValueError(f"digit_chunking must be one of 'block3', 'single', 'greedy', got {digit_chunking!r}")
        if preset == "code":
            split_digits = False
            hex_literals = True
            digit_chunk_size = 3
        elif preset == "math":
            split_digits = True
            hex_literals = True
            digit_chunk_size = None
        elif preset == "llama3" or preset == "gpt4":
            split_digits = False
            hex_literals = True
            digit_chunk_size = 3

        self.space_char = space_char
        self.split_digits = split_digits
        self.split_punctuation = split_punctuation
        self.keep_special_tokens = keep_special_tokens
        self.special_token_pattern = special_token_pattern
        self.hex_literals = hex_literals
        self.digit_chunk_size = digit_chunk_size
        self.digit_chunking = digit_chunking
        self.preset = preset

        escaped_space = re.escape(self.space_char)
        special_token = special_token_pattern if self.keep_special_tokens else r"(?!x)x"
        url = r"https?://[a-zA-Z0-9][-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)"

        email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

        hashtag = rf"{escaped_space}?#\w+"
        mention = rf"{escaped_space}?@\w+"

        emoji = (
            r"(?:[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF])"
            r"(?:[\uFE0E\uFE0F])?"
            r"(?:[\U0001F3FB-\U0001F3FF])?"
            r"(?:\u200D(?:[\U0001F300-\U0001FAFF]|[\u2600-\u26FF]|[\u2700-\u27BF])(?:[\uFE0E\uFE0F])?(?:[\U0001F3FB-\U0001F3FF])?)*"
        )

        cjk = rf"{escaped_space}?[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]+"

        word = rf"{escaped_space}?[^\W\d_\s{escaped_space}]+(?:['’][^\W\d_\s{escaped_space}]+)*"

        # Code hexadecimal / binary literals (e.g. 0xDEADBEEF, 0b1010)
        hex_number = rf"{escaped_space}?0[xX][0-9a-fA-F]+|{escaped_space}?0[bB][01]+" if self.hex_literals else None

        # Issue #43: numbers are chunked into 1-3 digit blocks by default
        # (LLaMA-3 / GPT-4 style) so place-value arithmetic works per chunk.
        # Unicode-aware \d (== \p{Nd}) rather than [0-9]: ASCII-only [0-9]
        # would silently drop non-ASCII decimal digits (e.g. Arabic-Indic),
        # breaking roundtrip; \d matches identically in Python and Rust.
        if self.split_digits or self.digit_chunking == "single":
            number = rf"{escaped_space}?\d"
        elif self.digit_chunking == "greedy":
            number = rf"{escaped_space}?\d+"
        elif self.digit_chunk_size is not None and self.digit_chunk_size > 0:
            number = rf"{escaped_space}?\d{{1,{self.digit_chunk_size}}}"
        else:  # digit_chunking == "block3"
            number = rf"{escaped_space}?\d{{1,3}}"

        space_marker = rf"{escaped_space}+"
        whitespace = r"\s+"

        if self.split_punctuation:
            punctuation = rf"{escaped_space}?[^\w\s{escaped_space}]|{escaped_space}?_"
        else:
            punctuation = rf"{escaped_space}?[^\w\s{escaped_space}]+|{escaped_space}?_+"

        self.patterns = [
            special_token,
            url,
            email,
            hashtag,
            mention,
            emoji,
            cjk,
            word,
        ]
        if hex_number:
            self.patterns.append(hex_number)
        self.patterns.extend(
            [
                number,
                space_marker,
                whitespace,
                punctuation,
            ]
        )

        combined_pattern = "|".join(f"(?:{p})" for p in self.patterns)
        self.regex = _get_cached_regex(combined_pattern)

    def pre_tokenize_iter(
        self,
        text: str,
        alignment: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    ) -> Iterator[PreToken]:
        """Yields chunks with normalized and raw-text offsets."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if alignment is not None and len(alignment) != len(text):
            raise ValueError("alignment length must match normalized text length")

        # Issue #41: snap regex boundaries to extended grapheme clusters so no
        # boundary falls before a combining mark (\p{M}) or inside a ZWJ/virama
        # sequence. This is a faithful port of the native Rust snapping, so the
        # Python fallback matches the Rust engine chunk-for-chunk.
        spans = [m.span() for m in self.regex.finditer(text)]
        for s, e in _snap_spans_to_graphemes(text, spans):
            if alignment is None:
                raw_span = (s, e)
            else:
                source_spans = [entry if isinstance(entry, tuple) else (entry, entry + 1) for entry in alignment[s:e]]
                if not source_spans:
                    continue
                raw_span = (
                    min(span[0] for span in source_spans),
                    max(span[1] for span in source_spans),
                )
            yield PreToken(text=text[s:e], start=s, end=e, raw_span=raw_span)

    @property
    def _native_pretok_parity(self) -> bool:
        """True when this config is exactly the native Rust pre-tokenizer regex.

        The native regex hardcodes: hex literals ON, unbounded digit runs, the
        default special-token pattern and the default metaspace char. Any other
        config must use the Python regex or the two would diverge.
        """
        return (
            self.space_char == "\u2581"
            and not self.split_digits
            and self.split_punctuation
            and self.keep_special_tokens
            and self.special_token_pattern == r"<\|[^\s|]+\|>"
            and self.hex_literals
            and self.digit_chunk_size is None
            and self.digit_chunking == "block3"
        )

    def pre_tokenize(self, text: str) -> List[str]:
        """
        Returns a flat list of pre-tokenized chunk strings.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        # ponytail: Rust pre_tokenize for default config; Python fallback exact
        if _HAS_RUST_NORM and self._native_pretok_parity:
            assert _caliper_core is not None
            try:
                return _caliper_core.rust_pre_tokenize(text)
            except (ImportError, AttributeError, ValueError):
                pass
        return [t.text for t in self.pre_tokenize_with_offsets(text)]

    def pre_tokenize_with_offsets(
        self,
        text: str,
        alignment: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    ) -> List[PreToken]:
        """Returns chunks with normalized and, when supplied, original spans."""
        return list(self.pre_tokenize_iter(text, alignment))

    def explain(self, text: str) -> None:
        """
        Diagnostic display showing how the text is sliced into chunks with character offsets.
        """
        tokens = self.pre_tokenize_with_offsets(text)
        print(f"\nInput: {text!r}")
        print("Tokens with Spans:")
        for idx, tok in enumerate(tokens):
            print(f"  {idx:>3}: {tok.text!r:<20} Span: {tok.span}")
        print(f"Total Chunks: {len(tokens)}\n")


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    normalizer = Normalizer()
    pre_tokenizer = RegexPreTokenizer(split_digits=False)

    samples = [
        "def compute_sum(a: int, b: int) -> int:\n    return a + b  # 100% precision",
        "Cost is $1,499.99 for iPhone 15 Pro (visit https://apple.com, or email dev@apple.com)!",
        "Emoji test: 👨‍👩‍👧‍👦 family and 👍🏽 thumbs up",
        "我喜欢自然语言处理 and नमस्ते दुनिया",
        "<|user|> Calculate 1.5e-10 + 42 = ? <|endoftext|>",
    ]

    for sample in samples:
        norm = normalizer.normalize(sample)
        tokens = pre_tokenizer.pre_tokenize_with_offsets(norm)
        print("=" * 70)
        print(f"ORIGINAL : {sample}")
        print(f"NORMALIZED: {norm}")
        print(f"CHUNKS   : {[t.text for t in tokens]}")
        print(f"OFFSETS  : {[t.span for t in tokens[:5]]} ... (total: {len(tokens)})")
