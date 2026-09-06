"""
UniqToken Production Command-Line Interface (CLI).

Provides unified command-line entry points:
- uniqtoken train: Train Unigram / SuperBPE models from corpus files.
- uniqtoken encode: Tokenize text inputs to subword tokens or integer IDs with metrics.
- uniqtoken decode: Reconstruct original text losslessly from token IDs.
- uniqtoken compare: Side-by-side color-coded token comparison across engines.
- uniqtoken benchmark: Run the multilingual empirical benchmark suite.
- uniqtoken eval-downstream: Run downstream LLM context efficiency evaluations.
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from benchmarks.benchmark_suite import TokenizerBenchmarkSuite
from benchmarks.downstream_eval import DownstreamEvaluator
from .cem_merger import CrossEntropyMerging
from .indentation_compressor import IndentationCompressor
from .tokenizer import CustomTokenizer


def _print_msg(msg: str) -> None:
    """Print a message via tqdm.write if available, preserving active progress bars."""
    if tqdm is not None and hasattr(tqdm, "write"):
        tqdm.write(msg)
    else:
        print(msg)


def _reconfigure_stdio() -> None:
    """Force UTF-8 on stdio so non-ASCII text survives piped stdin/stdout (Windows)."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


def train_command(args: argparse.Namespace) -> int:
    """Handles 'uniqtoken train' to train Unigram and SuperBPE tokenizers from corpus files."""
    if args.vocab_size < 1:
        print("Error: --vocab-size must be a positive integer.", file=sys.stderr)
        return 1
    if args.superbpe_merges < 0:
        print("Error: --superbpe-merges must not be negative.", file=sys.stderr)
        return 1
    if args.script_temp is not None and args.script_temp <= 0:
        print("Error: --script-temp must be greater than zero.", file=sys.stderr)
        return 1
    if args.min_boundary_entropy is not None and args.min_boundary_entropy < 0:
        print("Error: --min-boundary-entropy must not be negative.", file=sys.stderr)
        return 1
    is_streaming = getattr(args, "streaming", False)
    chunk_size_mb = getattr(args, "chunk_size_mb", 500)
    if chunk_size_mb is None or chunk_size_mb <= 0:
        print("Error: --chunk-size-mb must be a positive integer.", file=sys.stderr)
        return 1
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    if is_streaming and args.superbpe_merges > 0:
        print(
            "Error: --streaming cannot be combined with --superbpe-merges (SuperBPE needs the full in-memory corpus).",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "no_progress", False):
        os.environ["UNIQTOKEN_NO_PROGRESS"] = "1"

    total_bytes = 0
    for path in args.corpus:
        p = Path(path)
        if not p.exists():
            print(f"Error: Corpus file not found: {path}", file=sys.stderr)
            return 1
        total_bytes += p.stat().st_size

    show_progress = (tqdm is not None) and not getattr(args, "no_progress", False)

    corpus: List[str] = []
    if not is_streaming:
        read_pbar = None
        if show_progress and total_bytes > 0:
            read_pbar = tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Reading/Processing corpus",
                dynamic_ncols=True,
                leave=True,
            )
        start_time = time.perf_counter()
        bytes_read = 0

        try:
            for path in args.corpus:
                p = Path(path)
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                doc_parts: List[str] = []
                with open(p, "rb") as f:
                    while True:
                        block = f.read(256 * 1024)
                        if not block:
                            break
                        bytes_read += len(block)
                        doc_parts.append(decoder.decode(block))
                        if read_pbar is not None:
                            read_pbar.update(len(block))
                            elapsed = time.perf_counter() - start_time
                            mb_per_sec = (bytes_read / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
                            read_pbar.set_postfix({"throughput": f"{mb_per_sec:.2f} MB/s"})

                doc_parts.append(decoder.decode(b"", final=True))
                document = "".join(doc_parts)
                if document:
                    # A corpus file is one document. Preserve indentation, blank lines,
                    # and trailing whitespace because they are meaningful training data.
                    corpus.append(document)
        finally:
            if read_pbar is not None:
                read_pbar.close()

    def _corpus_doc_stream():
        # One document per file, identical to non-streaming mode (whole-file
        # decode, not line-split), so streaming never changes training input.
        # Only one file is resident at a time: RAM stays bounded by the
        # largest single file instead of the whole corpus.
        for path in args.corpus:
            p = Path(path)
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            doc_parts: List[str] = []
            with open(p, "rb") as f:
                while True:
                    block = f.read(256 * 1024)
                    if not block:
                        break
                    doc_parts.append(decoder.decode(block))
            doc_parts.append(decoder.decode(b"", final=True))
            document = "".join(doc_parts)
            if document:
                yield document

    if is_streaming:
        if total_bytes == 0:
            print("Error: Corpus is empty.", file=sys.stderr)
            return 1
    elif not corpus:
        print("Error: Corpus is empty.", file=sys.stderr)
        return 1

    _print_msg(f"Training UniqToken tokenizer on {len(args.corpus)} corpus files (Target Vocab: {args.vocab_size})...")
    tok = CustomTokenizer.train_from_corpus(
        corpus=_corpus_doc_stream() if is_streaming else corpus,
        target_vocab_size=args.vocab_size,
        ranking_strategy=args.ranking_strategy,
        adaptive_multiplier=args.adaptive_multiplier,
        script_balance_temperature=args.script_temp,
        min_boundary_entropy=args.min_boundary_entropy,
        byte_fallback=not args.no_byte_fallback,
        split_digits=args.split_digits,
        hex_literals=not args.no_hex_literals,
        digit_chunk_size=args.digit_chunk_size,
        digit_chunking=args.digit_chunking,
        preset=args.preset,
        compress_indents=args.compress_indents,
        verbose=args.verbose,
        streaming=is_streaming,
        chunk_size_bytes=chunk_size_bytes,
    )

    if args.superbpe_merges > 0:
        _print_msg(f"Optimizing vocabulary with SuperBPE ({args.superbpe_merges} merges)...")
        pretok_chunks: List[str] = []
        for doc in corpus:
            if args.compress_indents:
                doc = IndentationCompressor.compress_indents(doc)
            norm = tok.normalizer.normalize(doc)
            pretok_chunks.extend(tok.pre_tokenizer.pre_tokenize(norm))
        cem = CrossEntropyMerging(max_merges=args.superbpe_merges, cross_word=True, verbose=args.verbose)
        sbp_model = cem.optimize(tok.model, chunks=pretok_chunks)
        tok = CustomTokenizer(
            normalizer=tok.normalizer,
            pre_tokenizer=tok.pre_tokenizer,
            model=sbp_model,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir))
    _print_msg(f"Saved trained tokenizer model to: {out_dir.resolve()} (Vocab size: {tok.vocab_size})")
    return 0


def _load_input(args: argparse.Namespace) -> str:
    """Reads --input (string or file path) or stdin; empty strings are honored."""
    if args.input is not None:
        p = Path(args.input)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        return args.input
    return sys.stdin.read()


def encode_command(args: argparse.Namespace) -> int:
    """Handles 'uniqtoken encode'."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model directory not found: {args.model}", file=sys.stderr)
        return 1

    try:
        tok = CustomTokenizer.load(str(model_path))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error: Failed to load model from {args.model}: {e}", file=sys.stderr)
        return 1

    text_input = _load_input(args)

    if args.with_metrics:
        report = tok.encode_with_metrics(text_input)
        metrics_payload = {
            "num_tokens": report.num_tokens,
            "num_bytes": report.num_bytes,
            "bytes_per_token": report.compression_ratio_bytes_per_token,
            "byte_fallback_rate": report.byte_fallback_rate,
            "tokens": report.tokens,
            "token_ids": report.token_ids,
            "token_spans": report.token_spans,
        }
        output_str = json.dumps(metrics_payload, indent=2, ensure_ascii=False)
    elif args.with_offsets:
        tokens_with_offsets = tok.encode_with_offsets(text_input)
        offsets_payload = [
            {"token": t.text, "id": t.id, "start": t.raw_span[0], "end": t.raw_span[1]} for t in tokens_with_offsets
        ]
        output_str = json.dumps(offsets_payload, indent=2, ensure_ascii=False)
    elif args.to_ids:
        token_ids = tok.encode_to_ids(text_input)
        output_str = json.dumps(token_ids) if args.json else " ".join(str(i) for i in token_ids)
    else:
        tokens = tok.encode(text_input)
        output_str = json.dumps(tokens, ensure_ascii=False) if args.json else " ".join(tokens)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_str + "\n")
    else:
        print(output_str)

    return 0


def _parse_token_ids(input_data: str) -> List[int]:
    """Parses either a JSON integer array or a whitespace-separated ID list."""
    if input_data.startswith("[") and input_data.endswith("]"):
        parsed = json.loads(input_data)
        if not isinstance(parsed, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in parsed
        ):
            raise ValueError("token ID list must contain only non-negative integers")
        return parsed

    token_ids: List[int] = []
    for part in input_data.split():
        if not part.isascii() or not part.isdigit():
            raise ValueError(f"invalid token ID {part!r}; expected a non-negative decimal integer")
        token_ids.append(int(part))
    return token_ids


def decode_command(args: argparse.Namespace) -> int:
    """Handles 'uniqtoken decode'."""
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model directory not found: {args.model}", file=sys.stderr)
        return 1

    try:
        tok = CustomTokenizer.load(str(model_path))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error: Failed to load model from {args.model}: {e}", file=sys.stderr)
        return 1

    input_data = _load_input(args).strip()

    try:
        token_ids = _parse_token_ids(input_data)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing token IDs: {e}", file=sys.stderr)
        return 1

    decoded = tok.decode(token_ids)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write(decoded)
    else:
        sys.stdout.write(decoded)
        sys.stdout.flush()

    return 0


COMPARE_BG_COLORS = (41, 42, 43, 44, 45, 46)
COMPARE_RESET = "\033[0m"
COMPARE_ENGINES = ("uniqtoken", "tiktoken")

# Small generic corpus so `compare` works without a saved model by training a throwaway demo tokenizer.
COMPARE_DEMO_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning and natural language processing",
    "def calculate_fibonacci(n: int) -> int:",
]


def _colorize_tokens(tokens: List[str], use_color: bool) -> str:
    """Renders tokens as `[pill]` segments with alternating ANSI background colors.

    ESC, CR, newline, and tab are shown escaped so token bytes can never inject
    terminal sequences through either rendering path.
    """
    pills: List[str] = []
    for index, token in enumerate(tokens):
        text = token.replace("\x1b", "\\x1b").replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
        if use_color:
            color = COMPARE_BG_COLORS[index % len(COMPARE_BG_COLORS)]
            pills.append(f"\033[{color}m[{text}]{COMPARE_RESET}")
        else:
            pills.append(f"[{text}]")
    return "".join(pills)


def _escape_token_bytes(raw: bytes) -> str:
    """Renders raw token bytes as text, escaping incomplete UTF-8 sequences instead of replacing them."""
    return raw.decode("utf-8", errors="backslashreplace")


def _tiktoken_tokens(text: str, encoding_name: str = "cl100k_base") -> Optional[List[str]]:
    """Tokenizes with tiktoken, returning None when the optional dependency is missing.

    Decodes each token's raw bytes: decoding single IDs as text would replace
    incomplete UTF-8 sequences with U+FFFD. Literal special-token text is treated
    as ordinary input.
    """
    try:
        import tiktoken
    except ImportError:
        return None
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(text, disallowed_special=())
    return [_escape_token_bytes(raw) for raw in encoding.decode_tokens_bytes(token_ids)]


def _demo_tokenizer(text: str) -> CustomTokenizer:
    """Trains a small throwaway tokenizer so `compare` works without `--model`."""
    corpus = list(COMPARE_DEMO_CORPUS)
    if text:
        corpus.append(text)
    previous_flag = os.environ.get("UNIQTOKEN_NO_PROGRESS")
    os.environ["UNIQTOKEN_NO_PROGRESS"] = "1"
    try:
        return CustomTokenizer.train_from_corpus(corpus, target_vocab_size=320, verbose=False)
    finally:
        if previous_flag is None:
            os.environ.pop("UNIQTOKEN_NO_PROGRESS", None)
        else:
            os.environ["UNIQTOKEN_NO_PROGRESS"] = previous_flag


def compare_command(args: argparse.Namespace) -> int:
    """Handles 'uniqtoken compare'."""
    text_input = _load_input(args)
    requested: List[str] = []
    for name in args.models.split(","):
        engine = name.strip().lower()
        if engine and engine not in requested:
            requested.append(engine)
    unknown = [engine for engine in requested if engine not in COMPARE_ENGINES]
    if unknown:
        print(
            f"Error: Unknown tokenizer engine(s): {', '.join(unknown)}. Supported: {', '.join(COMPARE_ENGINES)}",
            file=sys.stderr,
        )
        return 1
    if not requested:
        print("Error: --models must list at least one tokenizer engine.", file=sys.stderr)
        return 1

    use_color = not args.no_color
    results: List[Tuple[str, List[str]]] = []
    for engine in requested:
        if engine == "uniqtoken":
            if args.model:
                model_path = Path(args.model)
                if not model_path.exists():
                    print(f"Error: Model directory not found: {args.model}", file=sys.stderr)
                    return 1
                try:
                    tok = CustomTokenizer.load(str(model_path))
                except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Error: Failed to load model from {args.model}: {e}", file=sys.stderr)
                    return 1
            else:
                _print_msg("No --model given; training a small demo tokenizer for comparison...")
                tok = _demo_tokenizer(text_input)
            results.append(("UniqToken", tok.encode(text_input)))
        elif engine == "tiktoken":
            pieces = _tiktoken_tokens(text_input)
            if pieces is None:
                _print_msg("tiktoken is not installed; skipping tiktoken (pip install tiktoken).")
                continue
            results.append(("Tiktoken", pieces))

    if not results:
        print("Error: No tokenizer engines available to compare.", file=sys.stderr)
        return 1

    width = max(len(label) for label, _ in results)
    for label, tokens in results:
        print(f"{label:<{width}}: {_colorize_tokens(tokens, use_color)} ({len(tokens)} tokens)")

    if len(results) >= 2:
        best = min(results, key=lambda item: len(item[1]))
        worst = max(results, key=lambda item: len(item[1]))
        if len(best[1]) == len(worst[1]):
            print("Token counts are identical across engines.")
        else:
            saving = (len(worst[1]) - len(best[1])) / len(worst[1]) * 100
            print(f"Token Savings: +{saving:.1f}% fewer tokens with {best[0]}")

    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    """Handles 'uniqtoken benchmark'."""
    suite = TokenizerBenchmarkSuite()
    suite.print_summary_report(include_large_payloads=args.large_payloads)
    if args.export_markdown:
        suite.export_markdown_report(args.export_markdown)
        print(f"\n[Exporter] Saved Markdown report to: {args.export_markdown}")
    if args.export_latex:
        suite.export_latex_report(args.export_latex)
        print(f"\n[Exporter] Saved LaTeX report to: {args.export_latex}")
    return 0


def downstream_command(args: argparse.Namespace) -> int:
    """Handles 'uniqtoken eval-downstream'."""
    vs = 500 if args.smoke_test else args.vocab_size
    include_ext = not args.no_external and not args.smoke_test
    evaluator = DownstreamEvaluator(vocab_size=vs)
    results = evaluator.run_downstream_suite(include_external_baselines=include_ext)
    evaluator.print_report(results)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uniqtoken",
        description="UniqToken: Production-Grade Byte-Fallback Unigram Tokenizer Engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train tokenizer from text corpus")
    p_train.add_argument("--corpus", nargs="+", required=True, help="One or more text corpus files")
    p_train.add_argument("--vocab-size", type=int, default=8000, help="Target vocabulary size (default: 8000)")
    p_train.add_argument("--out", type=str, required=True, help="Directory to save trained model files")
    p_train.add_argument(
        "--ranking-strategy",
        choices=["char_savings", "byte_savings", "frequency", "pmi"],
        default="char_savings",
        help="Seed candidate ranking metric (default: char_savings)",
    )
    p_train.add_argument("--adaptive-multiplier", action="store_true", help="Adapt seed pool size to corpus entropy")
    p_train.add_argument("--script-temp", type=float, default=None, help="Script temperature balancing (e.g. 0.5)")
    p_train.add_argument("--min-boundary-entropy", type=float, default=None, help="Min branch entropy threshold")
    p_train.add_argument("--superbpe-merges", type=int, default=0, help="Post-training SuperBPE merge count")
    p_train.add_argument(
        "--preset",
        choices=["default", "code", "math", "llama3", "gpt4"],
        default=None,
        help="Pre-tokenization domain preset (default: None)",
    )
    p_train.add_argument("--split-digits", action="store_true", help="Split individual digits into discrete tokens")
    p_train.add_argument("--digit-chunk-size", type=int, default=None, help="Max digits per numeric token (e.g. 3)")
    p_train.add_argument(
        "--digit-chunking",
        choices=["block3", "single", "greedy"],
        default="block3",
        help="Digit chunking mode: block3 (default, 1-3 digits per token), single, or greedy (legacy)",
    )
    p_train.add_argument("--no-hex-literals", action="store_true", help="Disable hexadecimal/binary literal matching")
    p_train.add_argument("--compress-indents", action="store_true", help="Enable whitespace indentation compression")
    p_train.add_argument("--no-byte-fallback", action="store_true", help="Disable UTF-8 byte fallback")
    p_train.add_argument("--no-progress", action="store_true", help="Disable dynamic progress indicators")
    p_train.add_argument(
        "--streaming",
        action="store_true",
        help="Enable disk-backed external chunk counter for TB-scale out-of-core training",
    )
    p_train.add_argument(
        "--chunk-size-mb",
        type=int,
        default=500,
        help="In-memory chunk buffer size in MB before spilling to disk (default: 500)",
    )
    p_train.add_argument("-v", "--verbose", action="store_true", help="Verbose training progress output")
    p_train.set_defaults(func=train_command)

    # Encode
    p_encode = subparsers.add_parser("encode", help="Encode text to tokens or IDs")
    p_encode.add_argument("--model", type=str, required=True, help="Path to saved model directory")
    p_encode.add_argument("--input", type=str, default=None, help="Input string or path to text file")
    p_encode.add_argument("--out", type=str, default=None, help="Output file path (default: stdout)")
    output_group = p_encode.add_mutually_exclusive_group()
    output_group.add_argument("--to-ids", action="store_true", help="Output integer token IDs")
    output_group.add_argument("--with-offsets", action="store_true", help="Output tokens with exact character spans")
    output_group.add_argument("--with-metrics", action="store_true", help="Output diagnostic compression metrics")
    p_encode.add_argument("--json", action="store_true", help="Format output as JSON array")
    p_encode.set_defaults(func=encode_command)

    # Decode
    p_decode = subparsers.add_parser("decode", help="Decode token IDs back to text")
    p_decode.add_argument("--model", type=str, required=True, help="Path to saved model directory")
    p_decode.add_argument("--input", type=str, default=None, help="Input ID sequence or path to file")
    p_decode.add_argument("--out", type=str, default=None, help="Output file path (default: stdout)")
    p_decode.set_defaults(func=decode_command)

    # Compare
    p_compare = subparsers.add_parser("compare", help="Side-by-side token comparison across engines")
    p_compare.add_argument("--input", type=str, default=None, help="Input string or path to text file")
    p_compare.add_argument(
        "--models",
        type=str,
        default="uniqtoken,tiktoken",
        help="Comma-separated engines to compare (uniqtoken,tiktoken)",
    )
    p_compare.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to saved UniqToken model directory (trains a small demo tokenizer if omitted)",
    )
    p_compare.add_argument("--no-color", action="store_true", help="Disable ANSI colors in token pills")
    p_compare.set_defaults(func=compare_command)

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark suite")
    p_bench.add_argument("--large-payloads", action="store_true", help="Run large 1MB/10MB throughput tests")
    p_bench.add_argument("--export-markdown", type=str, default=None, help="Path to save Markdown report")
    p_bench.add_argument("--export-latex", type=str, default=None, help="Path to save LaTeX table")
    p_bench.set_defaults(func=benchmark_command)

    # Downstream Eval
    p_down = subparsers.add_parser("eval-downstream", help="Run downstream LLM context efficiency eval")
    p_down.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size for evaluation")
    p_down.add_argument("--smoke-test", action="store_true", help="Run quick verification smoke test")
    p_down.add_argument("--no-external", action="store_true", help="Skip querying external baseline packages")
    p_down.set_defaults(func=downstream_command)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _reconfigure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 - top-level guard converts unexpected errors to exit code 1
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
