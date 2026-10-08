"""Before/after NFKC parity and paired throughput using synthetic fixtures only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import unicodedata

from benchmarks.profile_boundary_overhead import capture, digest, paired_interval, snapshot, timed, timing_summary
from benchmarks.profile_hot_paths import FIXTURES, STAGES, git_value, make_tokenizer, sha256, tool_version

FLAGS = (
    "normalize_unicode",
    "normalize_unicode_spaces",
    "normalize_punctuation",
    "lowercase",
    "collapse_whitespaces",
    "strip_whitespace",
)
TEXTS = (
    "",
    "ASCII text 42 < 3 | 4",
    "".join(chr(i) for i in range(128)),
    "  A\t  B\r\n C \x1c\x85",
    "literal \u2581 and \ue000\ue001",
    "\u00a0\u1680\u2000\u2007\u202f\u205f\u3000",
    "\u201cHello\u201d \u2018world\u2019\u2014\u2212\u2026",
    "A\u030a Cafe\u0301",
    "q\u0301 a\u0315\u0300 \u0301\u0323",
    "\u00c5 Caf\u00e9 q\u0301",
    "\ufb01 \u00b2 \u212b \uff21\uff22\uff23",
    "\u1100\u1161\u11a8 \uac01",
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u4e16\u754c \u0627\u0644\u0639\u0631\u0628\u064a\u0629",
    "\U0001f469\u200d\U0001f4bb \U0001f44d\U0001f3fd \U0001f1ee\U0001f1f3",
    "\u0600A\r\n\ufe0f\u200c\u200d\U000e0067\U000e007f",
    "\uffff\U0010ffff\u0378\x00",
    "<|unk|> <|system|> <|",
    "\ufe64\uff5cunk\uff5c\uff1e \uff1c\uff5csystem\uff5c\uff1e",
    "<\u0301|unk|>",
    "surrogate \ud800\udfff",
)


def fixtures():
    return {
        **{name: text for name, text in FIXTURES.items() if name != "multilingual"},
        "normalized_multilingual": unicodedata.normalize("NFKC", FIXTURES["multilingual"]),
        "non_normalized_multilingual": FIXTURES["multilingual"],
        "compatibility_start": "\ufb01 " + FIXTURES["long"],
        "compatibility_end": FIXTURES["long"] + " \ufb01",
        "combining_end": FIXTURES["long"] + " A\u030a",
        "normalized_combining": "q\u0301 \U0001f469\u200d\U0001f4bb " * 24,
    }


def load_native(path):
    spec = importlib.util.spec_from_file_location(f"_nfkc_{sha256(path)[:16]}.uniqtoken_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("native extension loader unavailable")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


def bind(native):
    # Each binary has its own PyO3 class identities. Keep a separate model/trie
    # per variant and swap module aliases outside all timed regions.
    for name, module in list(sys.modules.items()):
        if name.startswith("uniqtoken."):
            for alias in ("_native_core", "uniqtoken_core", "_uniqtoken_core"):
                if hasattr(module, alias):
                    setattr(module, alias, native)


def parity(variants):
    from uniqtoken.pre_tokenizer import Normalizer

    hashes = []
    for native, tok in variants:
        bind(native)
        if tok._native_pipeline_kwargs() is None or tok.model._get_rust_trie() is None:
            raise RuntimeError("build the source checkout with maturin develop; native execution is required")
    for values in itertools.product((False, True), repeat=len(FLAGS)):
        records = []
        for native, tok in variants:
            bind(native)
            tok.normalizer = Normalizer(**dict(zip(FLAGS, values)))
            flags = tuple(values)
            normalizers = [
                capture(lambda t=text, c=space: native.rust_normalize_with_alignment(t, c, *flags))
                for space in ("\u2581", "@")
                for text in TEXTS
            ]
            token_only = [capture(lambda t=text: native.rust_normalize(t, "\u2581", *flags)) for text in TEXTS]
            public = [snapshot(tok, text, {}) for text in TEXTS]
            pretokens = [
                capture(lambda t=text: tok.pre_tokenizer.pre_tokenize(tok.normalizer.normalize(t))) for text in TEXTS
            ]
            records.append([normalizers, token_only, public, pretokens])
        if records[0] != records[1]:
            raise AssertionError(f"normalization/public API parity: {dict(zip(FLAGS, values))}")
        hashes.append(digest(records[0]))
    # Security policies remain independent of the normalize_unicode flag.
    for normalize_unicode, action in itertools.product((False, True), ("escape", "raise", "ignore", "allow")):
        records = []
        for native, tok in variants:
            bind(native)
            tok.normalizer = Normalizer(normalize_unicode=normalize_unicode)
            options = {"allowed_special": "all"} if action == "allow" else {"disallowed_special_action": action}
            records.append([snapshot(tok, text, options) for text in TEXTS[-4:]])
        if records[0] != records[1]:
            raise AssertionError(f"security policy parity: {normalize_unicode}/{action}")
        hashes.append(digest(records[0]))
    errors = []
    for native, tok in variants:
        bind(native)
        tok.normalizer = Normalizer()
        errors.append(
            [capture(lambda t=text: native.rust_normalize(t)) for text in (None, 1, b"\xff", [], "\ud800")]
            + [capture(lambda c=space: native.rust_normalize("ASCII", c)) for space in ("", "ab", "\ue000", "\ue001")]
            + [snapshot(tok, text, {}) for text in (None, 1, b"\xff", [])]
        )
    if errors[0] != errors[1]:
        raise AssertionError("unsupported-input/error parity")
    signatures = [
        {name: getattr(native, name).__text_signature__ for name in dir(native) if name.startswith("rust_")}
        for native, _ in variants
    ]
    if signatures[0] != signatures[1]:
        raise AssertionError("native signature parity")
    return {"configurations": 64, "texts": len(TEXTS), "output_sha256": digest(hashes + [errors[0]]), "misses": 0}


def measure(variants, repetitions, iterations, warmup):
    rows = []
    for name, text in fixtures().items():
        for batch in (1, 32):
            texts = [text] if batch == 1 else [f"{text} {index:02d}" for index in range(batch)]
            calls, normalized_bytes, outputs, stages = [], [], [], []
            for native, tok in variants:
                bind(native)
                calls.append(
                    (lambda t=tok: t.encode(texts[0])) if batch == 1 else (lambda t=tok: t.encode_batch(texts))
                )
                normalized_bytes.append(sum(len(tok.normalizer.normalize(t).encode("utf-8")) for t in texts))
                outputs.append(digest(calls[-1]()))
                kwargs, trie = tok._native_pipeline_kwargs(), tok.model._get_rust_trie()
                if kwargs is None or trie is None:
                    raise RuntimeError("fused native path required")
                samples = []
                for _ in range(7):
                    observed = native.rust_profile_native_batch(texts, trie, tok.model.byte_fallback, **kwargs)
                    samples.append([sum(row[2][i] for row in observed) / 1e6 for i in range(len(STAGES))])
                stages.append(dict(zip(STAGES, map(statistics.median, zip(*samples)))))
            if outputs[0] != outputs[1] or normalized_bytes[0] != normalized_bytes[1]:
                raise AssertionError(f"timing fixture parity: {name}/{batch}")
            for _ in range(warmup):
                for variant in (0, 1):
                    bind(variants[variant][0])
                    timed(calls[variant], iterations)
            samples = [[], []]
            for index in range(repetitions):
                for variant in (0, 1) if index % 2 == 0 else (1, 0):
                    bind(variants[variant][0])
                    samples[variant].append(timed(calls[variant], iterations))
            ratios = [after / before for before, after in zip(*samples)]
            rows.append(
                {
                    "name": f"{name}_{batch}",
                    "already_nfkc": unicodedata.normalize("NFKC", text) == text,
                    "ascii": text.isascii(),
                    "normalized_bytes": normalized_bytes[0],
                    "input_sha256": digest(texts),
                    "output_sha256": outputs[0],
                    "baseline": timing_summary(samples[0]),
                    "optimized": timing_summary(samples[1]),
                    "baseline_MB_s": normalized_bytes[0] / statistics.median(samples[0]) / 1000,
                    "optimized_MB_s": normalized_bytes[0] / statistics.median(samples[1]) / 1000,
                    "paired_ratio_median": statistics.median(ratios),
                    "paired_ratio_interval_95": paired_interval(ratios),
                    "stage_median_ms": {"baseline": stages[0], "optimized": stages[1]},
                }
            )
    return rows


def run(args):
    if args.repetitions < 5 or args.iterations < 1 or args.warmup < 1:
        raise ValueError("repetitions >= 5, iterations >= 1 and warmup >= 1 required")
    if args.output.exists() or git_value("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("publication requires a new output file and clean committed source")
    os.environ["RAYON_NUM_THREADS"] = "1"
    # Preflight metadata before the expensive gate/timings; retain raw samples.
    metadata = {
        "source_commit": git_value("rev-parse", "HEAD"),
        "baseline_commit": git_value("rev-parse", f"{args.baseline_commit}^{{commit}}"),
        "source_sha256": {
            name: hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{name}"])).hexdigest()
            for name in (
                "crates/uniqtoken_core/src/normalizer.rs",
                "crates/uniqtoken_core/src/pipeline.rs",
                "crates/uniqtoken_core/Cargo.lock",
                "pyproject.toml",
                "benchmarks/profile_nfkc_fast_path.py",
            )
        },
        "python": sys.version,
        "unicode": unicodedata.unidata_version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "rustc": tool_version("rustc", "--version"),
        "cargo": tool_version("cargo", "--version"),
        "build": "release; default features; no allocation profiler",
        "threads": 1,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "iterations": args.iterations,
        "native_sha256": {"baseline": sha256(args.before), "optimized": sha256(args.after)},
        "fixtures_sha256": digest(fixtures()),
        "timing": "alternating paired trials; alias binding outside timer; detection inside production calls",
        "uncertainty": "deterministic bootstrap of paired median; within-run only",
    }
    variants = []
    for path in (args.before, args.after):
        native = load_native(path)
        if hasattr(native, "rust_allocation_profile_native_batch"):
            raise RuntimeError("timing requires default builds without allocation profiling")
        bind(native)
        tok = make_tokenizer()
        tok.model._get_rust_trie()
        variants.append((native, tok))
    models = [digest([tok.model.vocab, tok.model.token_to_id]) for _, tok in variants]
    if models[0] != models[1]:
        raise AssertionError("different model between variants")
    metadata["model_sha256"] = models[0]
    metadata["parity"] = parity(variants)
    metadata["workloads"] = measure(variants, args.repetitions, args.iterations, args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "parity": metadata["parity"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=41)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    run(parser.parse_args())
