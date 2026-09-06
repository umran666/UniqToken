"""Matched-budget vocab quality race.

For a fixed target vocab budget, train or load the same number of
candidate tokenizers on the same corpus and feed each one through
:func:`benchmarks.train_toy_transformer.train_toy_transformer` so that
they can be compared on:

- bytes/token (compression)
- held-out validation cross-entropy
- bits-per-byte (BPB)
- tokens/sec and bytes/sec (training throughput; GPU is used when
  available — the underlying ``train_toy_transformer`` already
  dispatches via ``torch.cuda.is_available()``)

Trainable baselines (matched budget, apples-to-apples):
  - UniqToken Unigram (UniqToken PMI Unigram trainer)
  - UniqToken BPE     (UniqToken BPE trainer)
  - UniqToken SuperBPE (UniqToken Unigram + CEM cross-word merging)
  - SentencePiece Unigram (the real ``sentencepiece`` trainer,
                           imported into UniqToken for inference)

Pretrained baselines (fixed vocab, informational only):
  - tiktoken ``cl100k_base``  (~100k vocab, byte-level BPE)
  - HuggingFace GPT-2         (~50k vocab, byte-level BPE)

Why matched budget matters: speed and inference memory are
vocab-bounded, so the only fair comparison at "this deployment will
pay N tokens of headroom" is when the candidate vocabs are all of
size N. Pretrained baselines are reported separately because nobody
can re-train GPT-2's vocab to 500 pieces.

Usage:
    python -m benchmarks.vocab_quality_race --budget 400 --steps 8 \\
            --export-json reports/vocab_race_400.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from benchmarks.train_toy_transformer import (  # noqa: E402
    PRETRAINING_CORPUS,
    BPETokenizerAdapter,
    _split_documents,
    train_toy_transformer,
)


@dataclass
class RaceEntry:
    """One row of the matched-budget race report."""

    tokenizer: str
    category: str
    target_vocab: int
    actual_vocab: int
    trained_fresh: bool
    bytes_per_token: float
    evaluated_tokens: int
    evaluated_bytes: int
    final_loss: float
    bits_per_byte: float
    tokens_per_sec: float
    bytes_per_sec: float
    wallclock_sec: float
    notes: str = ""


@dataclass
class RaceReport:
    budget: int
    corpus_size_documents: int
    corpus_size_bytes: int
    steps: int
    seed: int
    entries: List[RaceEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "budget": self.budget,
            "corpus_size_documents": self.corpus_size_documents,
            "corpus_size_bytes": self.corpus_size_bytes,
            "steps": self.steps,
            "seed": self.seed,
            "entries": [asdict(e) for e in self.entries],
        }


class _ExternalWrapper:
    """Adapter that exposes a pretrained tokenizer (tiktoken/HF) in the
    shape the harness expects: an integer ``vocab_size`` property and
    an ``encode_to_ids(text) -> List[int]`` method that does *not* add
    specials.
    """

    def __init__(
        self,
        name: str,
        vocab_size: int,
        encode_fn: Callable[[str], List[int]],
    ) -> None:
        self.name = name
        self._vocab_size = vocab_size
        self._encode = encode_fn

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode_to_ids(self, text: str) -> List[int]:
        return self._encode(text)


def _train_uniqtoken_unigram(budget: int, corpus: List[str]):
    """UniqToken PMI Unigram trained from scratch on the corpus."""
    from uniqtoken.tokenizer import CustomTokenizer

    return CustomTokenizer.train_from_corpus(
        corpus=corpus,
        target_vocab_size=budget,
        ranking_strategy="pmi",
        min_frequency=1,
        verbose=False,
    )


def _train_uniqtoken_bpe(budget: int, normalizer, pre_tokenizer, corpus: List[str]):
    """UniqToken BPE trained from scratch on the pre-tokenized corpus chunks.

    The harness's existing ``create_tokenizers`` already builds this; we
    repeat the logic here so the orchestrator owns the budget and can
    swap it independently of the legacy harness's hardcoded 500.
    """
    import uniqtoken.bpe_trainer as bpe_trainer

    chunks: List[str] = []
    for doc in corpus:
        norm = normalizer.normalize(doc)
        chunks.extend(pre_tokenizer.pre_tokenize(norm))
    trainer = bpe_trainer.BPETrainer(
        target_vocab_size=budget,
        byte_fallback=True,
    )
    model = trainer.train(chunks, verbose=False)
    return BPETokenizerAdapter(
        model,
        normalizer=normalizer,
        pre_tokenizer=pre_tokenizer,
    )


def _train_uniqtoken_superbpe(budget: int, corpus: List[str]):
    """UniqToken Unigram + CEM cross-word merging (SuperBPE)."""
    from uniqtoken.cem_merger import CrossEntropyMerging
    from uniqtoken.tokenizer import CustomTokenizer

    base = CustomTokenizer.train_from_corpus(
        corpus=corpus,
        target_vocab_size=budget,
        ranking_strategy="pmi",
        min_frequency=1,
        verbose=False,
    )
    chunks: List[str] = []
    for doc in corpus:
        norm = base.normalizer.normalize(doc)
        chunks.extend(base.pre_tokenizer.pre_tokenize(norm))
    merge_budget = max(0, budget - len(base.model.vocab))
    cem = CrossEntropyMerging(max_merges=min(30, merge_budget), cross_word=True, verbose=False)
    sbp_model = cem.optimize(base.model, chunks=chunks)
    return CustomTokenizer(
        normalizer=base.normalizer,
        pre_tokenizer=base.pre_tokenizer,
        model=sbp_model,
    )


def _train_sentencepiece(budget: int, corpus: List[str]):
    """Train a SentencePiece Unigram model at the matched budget, then
    import it into UniqToken for inference so the harness sees a uniform
    ``encode_to_ids`` interface.

    SPM is allowed to shrink the requested vocab to its observed
    vocabulary floor; we report the actual vocab size in the row.
    """
    import sentencepiece as spm
    from uniqtoken.sentencepiece_importer import import_sentencepiece

    with tempfile.TemporaryDirectory() as t:
        cpath = os.path.join(t, "c.txt")
        prefix = os.path.join(t, "sp")
        with open(cpath, "w", encoding="utf-8") as f:
            f.write("\n".join(corpus))
        target = budget
        for _ in range(5):
            try:
                spm.SentencePieceTrainer.Train(
                    input=cpath,
                    model_prefix=prefix,
                    vocab_size=target,
                    model_type="unigram",
                    character_coverage=0.9999,
                    byte_fallback=True,
                    normalization_rule_name="nfkc",
                    pad_id=0,
                    unk_id=1,
                    bos_id=-1,
                    eos_id=-1,
                    pad_piece="<pad>",
                    unk_piece="<unk>",
                )
                break
            except RuntimeError as exc:
                if "Vocabulary size too high" not in str(exc):
                    raise
                target = max(64, target - 25)
        else:
            raise RuntimeError(f"SentencePiece could not fit a Unigram at any budget (started at {budget})")
        sp = spm.SentencePieceProcessor()
        sp.Load(prefix + ".model")
        actual_size = sp.GetPieceSize()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = import_sentencepiece(prefix + ".model")
    setattr(cal, "_spm_actual_vocab", actual_size)
    return cal


def _load_tiktoken(name: str = "cl100k_base") -> _ExternalWrapper:
    import tiktoken

    enc = tiktoken.get_encoding(name)
    return _ExternalWrapper(
        name=f"tiktoken ({name})",
        vocab_size=enc.n_vocab,
        encode_fn=lambda text: enc.encode(text, disallowed_special=()),
    )


def _load_hf_gpt2() -> _ExternalWrapper:
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    return _ExternalWrapper(
        name="HuggingFace (GPT-2)",
        vocab_size=hf.vocab_size,
        encode_fn=lambda text: hf.encode(text, add_special_tokens=False),
    )


def run_vocab_quality_race(
    budget: int = 500,
    steps: int = 8,
    seed: int = 42,
    include_pretrained: bool = True,
    include_sentencepiece: bool = True,
    include_hf: Optional[bool] = None,
    include_tiktoken: Optional[bool] = None,
    train_kwargs: Optional[Dict[str, Any]] = None,
    device: str = "auto",
) -> RaceReport:
    """Train / load every candidate tokenizer at ``budget`` and produce
    a :class:`RaceReport`.

    Trainable baselines are always run. Pretrained baselines are
    attempted when ``include_pretrained`` (or the finer
    ``include_tiktoken`` / ``include_hf`` flags) is true; missing
    optional packages are skipped with a note in the report rather
    than failing the whole run.

    Defaults: ``include_pretrained=True`` enables *both* tiktoken and
    HF. The fine-grained flags default to ``None`` (inherit from
    ``include_pretrained``) — pass an explicit True/False to override.
    Set ``include_hf=False`` if your local environment has a broken
    ``transformers`` build (see
    https://github.com/huggingface/transformers/issues).

    The mini-transformer under the hood dispatches to GPU automatically
    when ``torch.cuda.is_available()`` is true. ``train_kwargs`` are
    forwarded to :func:`train_toy_transformer`; pass
    ``train_kwargs={"device": "cuda"}`` (or ``"cpu"``) to override
    on a per-call basis, or use the top-level ``device`` argument to
    set it for the whole race.
    """
    import torch  # noqa: F401  (used inside train_toy_transformer)

    if include_hf is None:
        include_hf = include_pretrained
    if include_tiktoken is None:
        include_tiktoken = include_pretrained

    train_kwargs = dict(train_kwargs or {})
    train_kwargs.setdefault("device", device)
    train_kwargs.setdefault("seed", seed)
    corpus_bytes = sum(len(d.encode("utf-8")) for d in PRETRAINING_CORPUS)
    train_corpus, _ = _split_documents(PRETRAINING_CORPUS)
    report = RaceReport(
        budget=budget,
        corpus_size_documents=len(PRETRAINING_CORPUS),
        corpus_size_bytes=corpus_bytes,
        steps=steps,
        seed=seed,
    )

    unigram_tok = _train_uniqtoken_unigram(budget, train_corpus)
    report.entries.append(_race_entry("UniqToken (Unigram)", "uniqtoken", budget, unigram_tok, steps, **train_kwargs))
    unigram_norm = unigram_tok.normalizer
    unigram_pretok = unigram_tok.pre_tokenizer

    bpe_tok = _train_uniqtoken_bpe(budget, unigram_norm, unigram_pretok, train_corpus)
    report.entries.append(_race_entry("UniqToken (BPE)", "uniqtoken", budget, bpe_tok, steps, **train_kwargs))

    sbp_tok = _train_uniqtoken_superbpe(budget, train_corpus)
    report.entries.append(_race_entry("UniqToken (SuperBPE)", "uniqtoken", budget, sbp_tok, steps, **train_kwargs))

    if include_sentencepiece:
        try:
            spm_tok = _train_sentencepiece(budget, train_corpus)
            actual = getattr(spm_tok, "_spm_actual_vocab", len(spm_tok.model.vocab))
            entry = _race_entry(
                "SentencePiece (Unigram)",
                "external_trainable",
                budget,
                spm_tok,
                steps,
                **train_kwargs,
            )
            entry.actual_vocab = actual
            entry.notes = f"SPM trained vocab capped at {actual} due to observed piece count" if actual < budget else ""
            report.entries.append(entry)
        except Exception as exc:
            warnings.warn(f"SentencePiece baseline unavailable ({exc}); skipping")

    want_pretrained = include_pretrained or include_tiktoken or include_hf
    if want_pretrained:
        if include_pretrained or include_tiktoken:
            try:
                tt = _load_tiktoken()
                entry = _race_entry(
                    tt.name,
                    "external_pretrained",
                    budget,
                    tt,
                    steps,
                    **train_kwargs,
                )
                entry.notes = "pretrained; vocab not matched to budget"
                report.entries.append(entry)
            except Exception as exc:
                warnings.warn(f"tiktoken baseline unavailable ({exc}); skipping")

        if include_pretrained or include_hf:
            try:
                gpt2 = _load_hf_gpt2()
                entry = _race_entry(
                    gpt2.name,
                    "external_pretrained",
                    budget,
                    gpt2,
                    steps,
                    **train_kwargs,
                )
                entry.notes = "pretrained; vocab not matched to budget"
                report.entries.append(entry)
            except Exception as exc:
                warnings.warn(f"HuggingFace GPT-2 baseline unavailable ({exc}); skipping")

    return report


def _race_entry(
    name: str,
    category: str,
    target_vocab: int,
    tok: Any,
    steps: int,
    **train_kwargs: Any,
) -> RaceEntry:
    t0 = time.perf_counter()
    m = train_toy_transformer(
        tok=tok,
        model_label=name,
        corpus=PRETRAINING_CORPUS,
        steps=steps,
        **train_kwargs,
    )
    elapsed = time.perf_counter() - t0
    return RaceEntry(
        tokenizer=name,
        category=category,
        target_vocab=target_vocab,
        actual_vocab=int(m.vocab_size),
        trained_fresh=(category != "external_pretrained"),
        bytes_per_token=float(m.compression_ratio),
        evaluated_tokens=int(m.evaluated_tokens),
        evaluated_bytes=int(m.evaluated_bytes),
        final_loss=float(m.final_loss),
        bits_per_byte=float(m.bits_per_byte),
        tokens_per_sec=float(m.tokens_per_sec),
        bytes_per_sec=float(m.bytes_per_sec),
        wallclock_sec=elapsed,
    )


def print_report(report: RaceReport, device: str = "auto") -> None:
    try:
        import torch

        actual = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        actual = "cpu (no torch)"
    print("=" * 110)
    print(
        f"MATCHED-BUDGET VOCAB QUALITY RACE  (budget={report.budget}, "
        f"steps={report.steps}, seed={report.seed}, device={device} -> {actual})"
    )
    print("=" * 110)
    header = (
        f"{'Tokenizer':<26} | {'Cat':<18} | {'Vocab':<6} | "
        f"{'Bytes/Tok':<10} | {'Loss':<8} | {'BPB':<8} | {'Tok/Sec':<10}"
    )
    print(header)
    print("-" * len(header))
    for e in report.entries:
        print(
            f"{e.tokenizer:<26} | {e.category:<18} | {e.actual_vocab:<6d} | "
            f"{e.bytes_per_token:<10.3f} | {e.final_loss:<8.4f} | "
            f"{e.bits_per_byte:<8.4f} | {e.tokens_per_sec:<10.1f}"
        )
    print("=" * 110)
    if any(e.notes for e in report.entries):
        print("\nNotes:")
        for e in report.entries:
            if e.notes:
                print(f"  - {e.tokenizer}: {e.notes}")


def write_json_report(report: RaceReport, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--budget", type=int, default=500, help="Matched vocab budget (default: 500)")
    parser.add_argument("--steps", type=int, default=8, help="Training steps for the mini-transformer (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Seed passed to the training loop (default: 42)")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip tiktoken and HF GPT-2 baselines")
    parser.add_argument(
        "--no-tiktoken", action="store_true", help="Skip tiktoken baseline (when using --no-pretrained overrides this)"
    )
    parser.add_argument(
        "--no-hf", action="store_true", help="Skip HuggingFace GPT-2 baseline (some Windows builds crash on import)"
    )
    parser.add_argument("--no-sentencepiece", action="store_true", help="Skip the SentencePiece Unigram baseline")
    parser.add_argument("--export-json", type=str, default=None, help="Path to write the JSON report")
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Compute device for the mini-transformer (default: auto)",
    )
    args = parser.parse_args()

    if args.budget < 64:
        parser.error("--budget must be at least 64 (UniqToken floor ~401 with byte_fallback)")
    if args.steps < 1:
        parser.error("--steps must be positive")

    report = run_vocab_quality_race(
        budget=args.budget,
        steps=args.steps,
        seed=args.seed,
        include_pretrained=not args.no_pretrained,
        include_tiktoken=not args.no_pretrained and not args.no_tiktoken,
        include_hf=not args.no_pretrained and not args.no_hf,
        include_sentencepiece=not args.no_sentencepiece,
        device=args.device,
        train_kwargs={
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "dim": args.dim,
            "heads": args.heads,
            "layers": args.layers,
        },
    )
    print_report(report, device=args.device)
    if args.export_json:
        write_json_report(report, args.export_json)
        print(f"[Exporter] Race report saved to: {args.export_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
