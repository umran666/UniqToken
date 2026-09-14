"""Staged research protocol. No synthetic data or baseline substitution in this CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
import unicodedata

from benchmarks.ledger import provenance, validate_ledger
from benchmarks.run_matched_budget_eval import CausalMiniTransformer, LMArchConfig
from uniqtoken.bpe_model import BPEModel
from uniqtoken.bpe_trainer import BPETrainer
from uniqtoken.cem_merger import CrossEntropyMerging
from uniqtoken.pre_tokenizer import Normalizer
from uniqtoken.tokenizer import CustomTokenizer

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_SCHEMA = 5  # Phase A freezing, resume, and tokenizer-training accounting.
DATASET_MANIFEST_SCHEMA = 2
FLOP_ESTIMATOR_VERSION = "dense_matmul_forward_backward_v1"
BYTE_BUDGET_FIELD = "normalized_utf8_bytes"
BYTE_AUDIT_FIELD = "source_utf8_bytes"
COHORT = ("sp_unigram", "sp_bpe", "boundary_bpe", "uniq_unigram", "uniq_superbpe")
VOCABS = (16384, 32768, 65536)
SPECIALS = ("<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>")
SPECIAL_IDS = dict(zip(SPECIALS, range(4)))
REGIMES = ("flops", "bytes")
NORMALIZATION = "NFKC_unicode_spaces_v1"
SPM_TRAINING_CHUNK_CHARACTERS = 1024
SCREEN = LMArchConfig("screen_2L_128d_512ff_untied", 2, 128, 4, 512, 1, 0.001)
CONFIRM = LMArchConfig("confirm_12L_768d_3072ff_untied", 12, 768, 12, 3072, 1, 0.0003)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new_json(path, payload):
    # Exclusive creation prevents an accidental rerun from overwriting evidence.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, allow_nan=False)


def write_new_json_atomic(path, payload):
    """Publish a new JSON artifact atomically without replacing prior evidence."""
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem hard link atomically publishes the fully flushed inode
        # and fails if another writer has already claimed the evidence path.
        os.link(temporary, path)
        Path(temporary).unlink()
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def runtime_identity():
    source = {}
    for directory, suffix in (("uniqtoken", ".py"), ("benchmarks", ".py"), ("crates", ".rs")):
        for path in sorted((ROOT / directory).rglob("*" + suffix)):
            if not {"target", "legacy"}.intersection(path.parts):
                source[path.relative_to(ROOT).as_posix()] = file_hash(path)
    for filename in ("pyproject.toml", "crates/uniqtoken_core/Cargo.toml", "crates/uniqtoken_core/Cargo.lock"):
        if (ROOT / filename).is_file():
            source[filename] = file_hash(ROOT / filename)
    spec = importlib.util.find_spec("uniqtoken_core")
    binaries = [] if spec is None else [Path(spec.origin)]
    if spec is not None and spec.submodule_search_locations:
        binaries = [p for root in spec.submodule_search_locations for p in Path(root).rglob("*")]
    hashes = sorted(file_hash(p) for p in binaries if p.suffix in (".pyd", ".so", ".dll", ".dylib"))
    versions = {"python": platform.python_version()}
    for package in ("torch", "sentencepiece", "numpy", "regex"):
        versions[package] = importlib.metadata.version(package)
    return {
        **provenance(),
        "source_hash": digest(source),
        "extension_hash": digest(hashes) if hashes else None,
        "extension_status": "installed" if hashes else "unavailable",
        "versions": versions,
    }


def normalize(text):
    return Normalizer.UNICODE_SPACES.sub(" ", unicodedata.normalize("NFKC", text))


RESERVED_CORPUS_TEXT = re.compile(r"\x00|<\||[\ue000\ue001\u2581]")


def tokenizer_training_input(name):
    require(name in COHORT, "unsupported tokenizer")
    if name.startswith("sp_"):
        return {
            "unit": "contiguous_normalized_document_chunks",
            "maximum_unicode_characters": SPM_TRAINING_CHUNK_CHARACTERS,
            "preserves_normalized_utf8_bytes": True,
        }
    return {
        "unit": "normalized_documents",
        "maximum_unicode_characters": None,
        "preserves_normalized_utf8_bytes": True,
    }


def tokenizer_configuration(name, budget):
    """Ledger-visible Phase A training configuration for provenance validation."""
    require(name in COHORT and type(budget) is int and budget > 0, "invalid tokenizer configuration")
    common = {
        "normalization": NORMALIZATION,
        "vocab_size": budget,
        "special_tokens": SPECIAL_IDS,
        "byte_fallback": True,
        "training_input": tokenizer_training_input(name),
    }
    if name.startswith("sp_"):
        return {
            **common,
            "implementation": "sentencepiece",
            "model_type": name.removeprefix("sp_"),
            "hard_vocab_limit": True,
            "character_coverage": 1.0,
            "normalization_rule_name": "identity",
            "remove_extra_whitespaces": False,
            "add_dummy_prefix": False,
            "shuffle_input_sentence": False,
            "input_sentence_size": 0,
            "num_threads": 1,
        }
    if name == "boundary_bpe":
        return {**common, "implementation": "uniqtoken_boundary_bpe", "boundary_pattern": r"\S+|\s"}
    reserve = min(budget // 10, 4000) if name == "uniq_superbpe" else 0
    config = {
        **common,
        "implementation": "uniqtoken_unigram",
        "unigram_vocab_size": budget - reserve,
        "minimum_frequency": 1,
        "minimum_edge_log_probability": "negative_infinity",
        "native_em_fallback": False,
    }
    if reserve:
        config["cem"] = {"maximum_merges": reserve, "cross_word": True, "document_separator": SPECIALS[3]}
    return config


def sentencepiece_training_sentences(texts):
    """Bound SentencePiece training units without adding, removing, or reordering text."""
    for text in texts:
        for start in range(0, len(text), SPM_TRAINING_CHUNK_CHARACTERS):
            yield text[start : start + SPM_TRAINING_CHUNK_CHARACTERS]


def jsonl_rows(path):
    """Parse LF-delimited JSON without treating Unicode separators as record boundaries."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def load_dataset(manifest_path):
    """Read frozen JSONL splits ({id, text}); verify file and normalized-document identity."""
    path = Path(manifest_path)
    manifest = read_json(path)
    require(manifest.get("schema_version") == DATASET_MANIFEST_SCHEMA, "unsupported dataset manifest schema")
    for key in ("dataset_id", "source", "license", "deduplication"):
        require(isinstance(manifest.get(key), str) and manifest[key].strip(), f"dataset requires {key}")
    require(manifest.get("normalization") == NORMALIZATION, "undeclared or incompatible normalization")
    freeze = manifest.get("freeze")
    require(isinstance(freeze, dict) and freeze.get("immutable") is True, "dataset manifest is not frozen")
    require(
        isinstance(freeze.get("source_files"), list) and freeze["source_files"], "frozen source-file inventory required"
    )
    require(
        isinstance(freeze.get("source_revisions"), dict)
        and freeze["source_revisions"]
        and all(isinstance(value, str) and value for value in freeze["source_revisions"].values()),
        "frozen source revisions required",
    )
    selection = freeze.get("selection")
    require(
        isinstance(selection, dict)
        and selection.get("byte_unit") == "MB_decimal"
        and isinstance(selection.get("groups"), list)
        and selection["groups"],
        "frozen selection accounting required",
    )
    source_inventory = {}
    for source_file in freeze["source_files"]:
        require(
            isinstance(source_file, dict)
            and all(
                isinstance(source_file.get(k), str) and source_file[k]
                for k in ("local_path", "sha256", "dataset", "revision", "url", "license", "release_variant")
            )
            and type(source_file.get("file_bytes")) is int
            and source_file["file_bytes"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", source_file["sha256"]) is not None
            and source_file["revision"].lower() not in {"main", "master", "latest"},
            "invalid pinned source-file inventory",
        )
        local_source = path.parent / source_file["local_path"]
        require(
            local_source.is_file() and file_hash(local_source) == source_file["sha256"],
            "source file missing or hash mismatch",
        )
        require(local_source.stat().st_size == source_file["file_bytes"], "source file byte count mismatch")
        source_inventory[(source_file["dataset"], source_file["revision"], source_file["local_path"])] = source_file
    require(set(manifest.get("splits", {})) == {"train", "validation", "test"}, "three splits required")
    docs, assignment, document_bytes, seen_ids, seen_text = {}, {}, {}, set(), set()
    selected_groups = {}
    for split, entry in manifest["splits"].items():
        data_path = path.parent / entry["path"]
        require(file_hash(data_path) == entry.get("sha256"), f"{split} file hash mismatch")
        docs[split], assignment[split], document_bytes[split] = [], [], []
        for row in jsonl_rows(data_path):
            require(isinstance(row.get("id"), str) and row["id"], "document requires a nonempty string id")
            require(isinstance(row.get("text"), str) and row["text"].strip(), "empty document")
            require(isinstance(row.get("language"), str) and row["language"], "document requires language")
            require(isinstance(row.get("domain"), str) and row["domain"], "document requires domain")
            require(
                type(row.get("raw_utf8_bytes")) is int
                and type(row.get("normalized_utf8_bytes")) is int
                and row["raw_utf8_bytes"] >= 0
                and row["normalized_utf8_bytes"] >= 0,
                "document requires raw and normalized UTF-8 byte counts",
            )
            source = row.get("source")
            require(
                isinstance(source, dict)
                and all(
                    isinstance(source.get(k), str) and source[k]
                    for k in (
                        "dataset",
                        "revision",
                        "url",
                        "local_path",
                        "source_file_sha256",
                        "license",
                        "release_variant",
                    )
                )
                and source["revision"].lower() not in {"main", "master", "latest"}
                and re.fullmatch(r"[0-9a-f]{64}", source["source_file_sha256"]) is not None,
                "document requires pinned source provenance",
            )
            inventory_source = source_inventory.get((source["dataset"], source["revision"], source["local_path"]))
            require(
                inventory_source is not None
                and source["source_file_sha256"] == inventory_source["sha256"]
                and source["url"] == inventory_source["url"],
                "document source provenance is not in the frozen source inventory",
            )
            dedup = row.get("dedup")
            require(
                isinstance(dedup, dict) and dedup.get("status") == "accepted_after_exact_and_near_eval_check",
                "document lacks dedup status",
            )
            text = normalize(row["text"])
            text.encode("utf-8", errors="strict")
            require(
                row["raw_utf8_bytes"] == len(row["text"].encode("utf-8", errors="strict"))
                and row["normalized_utf8_bytes"] == len(text.encode("utf-8")),
                "document UTF-8 byte counts do not match text",
            )
            require(not RESERVED_CORPUS_TEXT.search(text), "reserved control/metaspace text in corpus")
            require(
                row["id"] not in seen_ids and digest(text) not in seen_text,
                "duplicate document or train/validation/test leakage",
            )
            seen_ids.add(row["id"])
            seen_text.add(digest(text))
            docs[split].append(text)
            assignment[split].append(
                {"id": row["id"], "normalized_sha256": digest(text), "source_sha256": digest(row["text"])}
            )
            document_bytes[split].append(
                {
                    BYTE_AUDIT_FIELD: len(row["text"].encode("utf-8", errors="strict")),
                    BYTE_BUDGET_FIELD: len(text.encode("utf-8")),
                }
            )
            group_key = (split, source["dataset"], source["release_variant"], row["language"], row["domain"])
            group = selected_groups.setdefault(
                group_key,
                {
                    "split": split,
                    "dataset": source["dataset"],
                    "release_variant": source["release_variant"],
                    "language": row["language"],
                    "domain": row["domain"],
                    "documents": 0,
                    "raw_utf8_bytes": 0,
                    "normalized_utf8_bytes": 0,
                },
            )
            group["documents"] += 1
            group["raw_utf8_bytes"] += row["raw_utf8_bytes"]
            group["normalized_utf8_bytes"] += row["normalized_utf8_bytes"]
        require(docs[split], f"empty {split} split")
    require(
        selection["groups"] == [selected_groups[key] for key in sorted(selected_groups)],
        "frozen selection accounting does not match split contents",
    )
    dataset = {"manifest": manifest, "assignment_hash": digest(assignment), "manifest_hash": digest(manifest)}
    dataset["document_bytes"] = document_bytes
    dataset["splits"] = {
        split: {
            "documents": len(texts),
            "utf8_bytes": sum(len(t.encode("utf-8")) for t in texts),
            **byte_totals(document_bytes[split]),
            "unicode_characters": sum(map(len, texts)),
        }
        for split, texts in docs.items()
    }
    return docs, dataset


@dataclass
class ResearchTokenizer:
    name: str
    model: object
    vocab: dict
    merges: int = 0

    def encode(self, text):
        if self.name.startswith("sp_"):
            ids = list(self.model.encode(text, out_type=int))
        elif self.name == "boundary_bpe":
            # Preserve every whitespace character, but never learn/apply merges across it.
            ids = [i for chunk in re.findall(r"\S+|\s", text) for i in self.model.encode_to_ids(chunk)]
        else:
            ids = self.model.encode_to_ids(text)
        require(
            ids and all(type(i) is int and 4 <= i < len(self.vocab) for i in ids),
            "empty encoding, unknown/control token, or out-of-range ID",
        )
        require(self.model.decode(ids) == text, f"{self.name}: normalized roundtrip failure")
        return ids

    def piece_for_id(self, token_id):
        if self.name.startswith("sp_") and hasattr(self.model, "id_to_piece"):
            return self.model.id_to_piece(token_id)
        id_to_token = getattr(self.model, "id_to_token", None)
        if id_to_token is None:
            id_to_token = getattr(getattr(self.model, "model", None), "id_to_token", None)
        require(id_to_token is not None, f"{self.name}: model has no ID-to-token mapping")
        return id_to_token[token_id]


def validate_tokenizer(tok, budget):
    require(
        len(tok.vocab) == budget and set(tok.vocab.values()) == set(range(budget)),
        "exact vocabulary budget/ID invariant failed",
    )
    require(all(tok.vocab.get(t) == i for t, i in SPECIAL_IDS.items()), "special-token accounting mismatch")
    require(all(f"<0x{i:02X}>" in tok.vocab for i in range(256)), "all 256 byte tokens required")
    require(tok.name != "uniq_superbpe" or tok.merges > 0, "SuperBPE zero-merge condition")


def train_tokenizer(name, texts, budget, directory):
    require(name in COHORT, "unsupported tokenizer; substitution is forbidden")
    directory.mkdir()  # No reuse of a model from an earlier condition.
    started = time.perf_counter()
    if name.startswith("sp_"):
        import sentencepiece as spm

        spm.SentencePieceTrainer.train(
            sentence_iterator=sentencepiece_training_sentences(texts),
            model_prefix=str(directory / "sp"),
            model_type=name.removeprefix("sp_"),
            vocab_size=budget,
            hard_vocab_limit=True,
            character_coverage=1.0,
            byte_fallback=True,
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
            add_dummy_prefix=False,
            unk_id=0,
            pad_id=1,
            bos_id=2,
            eos_id=3,
            unk_piece=SPECIALS[0],
            pad_piece=SPECIALS[1],
            bos_piece=SPECIALS[2],
            eos_piece=SPECIALS[3],
            shuffle_input_sentence=False,
            input_sentence_size=0,
            num_threads=1,
            minloglevel=2,
            max_sentence_length=4 * SPM_TRAINING_CHUNK_CHARACTERS + 1,
        )
        model = spm.SentencePieceProcessor(model_file=str(directory / "sp.model"))
        tok = ResearchTokenizer(name, model, {model.id_to_piece(i): i for i in range(model.vocab_size())})
    elif name == "boundary_bpe":
        model = BPETrainer(target_vocab_size=budget, special_tokens=list(SPECIALS), byte_fallback=True).train(
            [c for t in texts for c in re.findall(r"\S+|\s", t)]
        )
        write_new_json(
            directory / "bpe.json",
            {"vocab": model.token_to_id, "merges": [[a, b, rank] for (a, b), rank in model.merges.items()]},
        )
        tok = ResearchTokenizer(name, model, model.token_to_id)
    else:
        reserve = min(budget // 10, 4000) if name == "uniq_superbpe" else 0
        model = CustomTokenizer.train_from_corpus(
            texts,
            target_vocab_size=budget - reserve,
            special_tokens=list(SPECIALS),
            byte_fallback=True,
            min_frequency=1,
            min_edge_log_prob=float("-inf"),
            verbose=False,
        )
        # Fresh experiment artifacts have no LM weights yet. Canonicalize control IDs
        # here, without changing pieces, probabilities, segmentation, or library APIs.
        ordered = [*SPECIALS, *(t for t in model.model.token_to_id if t not in SPECIAL_IDS)]
        model.model.token_to_id = {t: i for i, t in enumerate(ordered)}
        model.model.id_to_token = {i: t for i, t in enumerate(ordered)}
        validate_tokenizer(ResearchTokenizer("uniq_unigram", model, model.model.token_to_id), budget - reserve)
        merges = 0
        if reserve:
            cem = CrossEntropyMerging(max_merges=reserve, cross_word=True)
            # EOS separates training documents and is non-mergeable in the existing CEM implementation.
            chunks = [
                c
                for t in texts
                for c in [*model.pre_tokenizer.pre_tokenize(model.normalizer.normalize(t)), SPECIALS[3]]
            ]
            model.model = cem.optimize(model.model, chunks)
            merges = len(cem.merges)
        tok = ResearchTokenizer(name, model, model.model.token_to_id, merges)
        model.save(directory, save_binary=False)
    validate_tokenizer(tok, budget)
    elapsed = time.perf_counter() - started
    require(elapsed > 0.0, "nonpositive tokenizer training elapsed time")
    return tok, elapsed


def artifact_hashes(directory):
    return {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def load_tokenizer(row, root):
    directory = root / row["artifact"]
    require(directory.resolve().is_relative_to(root.resolve()), "artifact path escapes phase directory")
    require(artifact_hashes(directory) == row["artifact_hashes"], "stale or incomplete tokenizer artifacts")
    name = row["tokenizer"]
    if name.startswith("sp_"):
        import sentencepiece as spm

        model = spm.SentencePieceProcessor(model_file=str(directory / "sp.model"))
        tok = ResearchTokenizer(name, model, {model.id_to_piece(i): i for i in range(model.vocab_size())})
    elif name == "boundary_bpe":
        data = read_json(directory / "bpe.json")
        vocab = data["vocab"]
        model = BPEModel(
            set(vocab),
            vocab,
            {i: t for t, i in vocab.items()},
            {(a, b): r for a, b, r in data["merges"]},
            list(SPECIALS),
            True,
        )
        tok = ResearchTokenizer(name, model, vocab)
    else:
        model = CustomTokenizer.load(directory, prefer_binary=False)
        tok = ResearchTokenizer(name, model, model.model.token_to_id, row["learned_merges"])
    validate_tokenizer(tok, row["vocab_budget"])
    return tok


def byte_totals(document_bytes, count=None):
    """Sum an ordered document prefix, cycling exactly as the training scheduler does."""
    require(bool(document_bytes), "byte accounting requires documents")
    count = len(document_bytes) if count is None else count
    require(type(count) is int and count >= 0, "invalid document exposure count")
    cycles, tail = divmod(count, len(document_bytes))
    return {
        key: cycles * sum(d[key] for d in document_bytes) + sum(d[key] for d in document_bytes[:tail])
        for key in (BYTE_BUDGET_FIELD, BYTE_AUDIT_FIELD)
    }


def token_metrics(tok, texts, *, source_utf8_bytes):
    ids = [token_id for text in texts for token_id in tok.encode(text)]
    count = len(ids)
    byte_count = sum(len(t.encode("utf-8")) for t in texts)
    chars = sum(len(t) for t in texts)
    return {
        "tokens": count,
        "utf8_bytes": byte_count,
        BYTE_BUDGET_FIELD: byte_count,
        BYTE_AUDIT_FIELD: source_utf8_bytes,
        "unicode_characters": chars,
        "tokens_per_unicode_character": count / chars,
        "bytes_per_token": byte_count / count,
        "byte_fallback_tokens": sum(
            bool(re.fullmatch(r"<0x[0-9A-F]{2}>", tok.piece_for_id(token_id))) for token_id in ids
        ),
        "byte_fallback_percent": 100.0
        * sum(bool(re.fullmatch(r"<0x[0-9A-F]{2}>", tok.piece_for_id(token_id))) for token_id in ids)
        / count,
    }


def parameter_count(vocab, cfg, context):
    counts = parameter_accounting(vocab, cfg, context)
    return counts["total_params"], counts["non_embedding_params"]


def parameter_accounting(vocab, cfg, context):
    # Untied embeddings/head; PyTorch encoder biases, norms, learned positions and final norm.
    d = cfg.d_model
    non_embedding = cfg.num_layers * (4 * d * d + 2 * d * cfg.d_ff + cfg.d_ff + 9 * d) + context * d + 2 * d
    return {
        "core_params": non_embedding,
        "non_embedding_params": non_embedding,
        "input_embedding_params": vocab * d,
        "output_head_params": vocab * d,
        "total_params": non_embedding + 2 * vocab * d,
    }


def model_config(phase, device):
    cfg, context = (SCREEN, 128) if phase == "B" else (CONFIRM, 1024)
    return {
        "architecture": asdict(cfg),
        "context": context,
        "dtype": "float32",
        "device": device,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "dropout": 0.1,
        "activation": "relu",
        "position": "learned",
        "head": "untied",
        "batch_size": 1,
        "evaluation": "BOS_text_EOS_nonoverlapping_causal_windows",
        "flop_estimator": FLOP_ESTIMATOR_VERSION,
        "flop_tolerance": 0.01,
    }


def training_flops(vocab, cfg, length):
    """Dense matmul forward+backward estimate, not hardware FLOPs; excludes lookup/optimizer/norms."""
    return flop_accounting(vocab, cfg, length, length * length)["actual_analytical_flops"]


def flop_accounting(vocab, cfg, target_tokens, sequence_length_squared_sum):
    """Additive accounting across all executed windows; the estimator itself is unchanged."""
    d = cfg.d_model
    core = (
        6 * cfg.num_layers * target_tokens * (4 * d * d + 2 * d * cfg.d_ff)
        + 12 * cfg.num_layers * sequence_length_squared_sum * d
    )
    output = 6 * target_tokens * d * vocab
    return {
        "core_analytical_flops": core,
        "output_projection_flops": output,
        "actual_analytical_flops": core + output,
        "flop_estimator_version": FLOP_ESTIMATOR_VERSION,
    }


def windows(ids, context):
    # Predict all text tokens and EOS, including the first token from BOS. No dropped tail.
    sequence = [SPECIAL_IDS[SPECIALS[2]], *ids, SPECIAL_IDS[SPECIALS[3]]]
    for start in range(0, len(sequence) - 1, context):
        end = min(start + context, len(sequence) - 1)
        yield sequence[start:end], sequence[start + 1 : end + 1]


def nll_metrics(nll, targets, byte_count, *, source_utf8_bytes):
    require(math.isfinite(nll) and nll >= 0 and targets > 0 and byte_count > 0, "invalid held-out NLL totals")
    ce = nll / targets
    perplexity = math.exp(ce) if ce < math.log(sys.float_info.max) else None
    return {
        "total_nll_nats": nll,
        "target_tokens_including_eos": targets,
        "utf8_bytes": byte_count,
        BYTE_BUDGET_FIELD: byte_count,
        BYTE_AUDIT_FIELD: source_utf8_bytes,
        "nll_nats_per_token": ce,
        "bits_per_byte": nll / (byte_count * math.log(2)),
        "token_perplexity": perplexity,
        "perplexity_overflow": perplexity is None,
    }


def evaluate(model, encoded, texts, context, device, *, source_utf8_bytes):
    import torch
    import torch.nn.functional as F

    model.eval()
    total, targets = 0.0, 0
    with torch.no_grad():
        for ids in encoded:
            for x, y in windows(ids, context):
                logits = model(torch.tensor([x], device=device))
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), torch.tensor(y, device=device), reduction="sum"
                )
                total += float(loss.double().item())
                targets += len(y)
    return nll_metrics(total, targets, sum(len(t.encode("utf-8")) for t in texts), source_utf8_bytes=source_utf8_bytes)


def byte_schedule(texts, target):
    """Exact shared document prefix; fail rather than rounding/truncating a document."""
    require(type(target) is int and target > 0, "byte budget must be a positive integer")
    sizes = [len(t.encode("utf-8")) for t in texts]
    count, used = 0, 0
    while used < target:
        used += sizes[count % len(sizes)]
        count += 1
    require(used == target, "byte budget must end at a document boundary (use multiples of training split bytes)")
    return count


def train_lm(tok, docs, cfg, context, regime, budget, seed, device, evaluate_test, *, document_bytes):
    import torch
    import torch.nn.functional as F

    require(regime in REGIMES, "unknown matching regime")
    require(math.isfinite(budget) and budget > 0, "invalid matching budget")
    require(set(document_bytes) == set(docs), "missing source byte accounting")
    for split, texts in docs.items():
        require(len(document_bytes[split]) == len(texts), "document byte accounting mismatch")
        require(
            all(
                d[BYTE_BUDGET_FIELD] == len(t.encode("utf-8"))
                and type(d[BYTE_AUDIT_FIELD]) is int
                and d[BYTE_AUDIT_FIELD] > 0
                for t, d in zip(texts, document_bytes[split])
            ),
            "invalid source/normalized byte accounting",
        )
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device.startswith("cuda"):
        require(torch.cuda.is_available(), "requested CUDA is unavailable")
    model = CausalMiniTransformer(len(tok.vocab), cfg, context).to(device=device, dtype=torch.float32)
    actual = sum(p.numel() for p in model.parameters())
    total, non_embedding = parameter_count(len(tok.vocab), cfg, context)
    require(actual == total, "architecture parameter count mismatch")
    counts = parameter_accounting(len(tok.vocab), cfg, context)
    require(model.embed.weight is not model.head.weight, "untied embeddings required")
    require(
        model.embed.weight.numel() == counts["input_embedding_params"]
        and model.head.weight.numel() == counts["output_head_params"]
        and actual - model.embed.weight.numel() - model.head.weight.numel() == counts["core_params"],
        "parameter component mismatch",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    encoded = {s: [tok.encode(t) for t in texts] for s, texts in docs.items() if s != "test" or evaluate_test}
    doc_limit = byte_schedule(docs["train"], budget) if regime == "bytes" else None
    steps = tokens = doc_index = completed_bytes = 0
    flops = 0.0
    sequence_length_squared_sum = 0
    while doc_limit is None or doc_index < doc_limit:
        index = doc_index % len(docs["train"])
        for x, y in windows(encoded["train"][index], context):
            cost = training_flops(len(tok.vocab), cfg, len(x))
            if regime == "flops" and flops + cost > budget:
                # A partial final training window improves budget resolution without overshooting.
                length = len(x)
                while length and flops + training_flops(len(tok.vocab), cfg, length) > budget:
                    length -= 1
                if not length:
                    require(
                        steps > 0 and flops >= budget * 0.99, "FLOP budget cannot be matched within 1%; increase budget"
                    )
                    break
                x, y = x[:length], y[:length]
                cost = training_flops(len(tok.vocab), cfg, length)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.tensor([x], device=device))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), torch.tensor(y, device=device))
            require(bool(torch.isfinite(loss)), "non-finite native/LM computation")
            loss.backward()
            optimizer.step()
            steps += 1
            tokens += len(y)
            sequence_length_squared_sum += len(y) ** 2
            flops += cost
            if regime == "flops" and flops >= budget * 0.99:
                break
        else:
            completed_bytes += len(docs["train"][index].encode("utf-8"))
            doc_index += 1
            continue
        break
    result = {
        **counts,
        "training_steps": steps,
        "training_target_tokens": tokens,
        "completed_document_bytes": completed_bytes,
        "completed_training_documents": doc_index,
        "training_byte_scope": "complete_document_prefix",
        "training_bytes": byte_totals(document_bytes["train"], doc_index),
        "training_sequence_length_squared_sum": sequence_length_squared_sum,
        **flop_accounting(len(tok.vocab), cfg, tokens, sequence_length_squared_sum),
        "validation": evaluate(
            model,
            encoded["validation"],
            docs["validation"],
            context,
            device,
            source_utf8_bytes=byte_totals(document_bytes["validation"])[BYTE_AUDIT_FIELD],
        ),
    }
    if evaluate_test:
        result["test"] = evaluate(
            model,
            encoded["test"],
            docs["test"],
            context,
            device,
            source_utf8_bytes=byte_totals(document_bytes["test"])[BYTE_AUDIT_FIELD],
        )
    return result


def condition_key(row):
    return row["tokenizer"], row["vocab_budget"], row["budget_regime"], row["seed"]


def validate_phase_a_condition_record(row, identity, dataset, expected_key, artifact_root=None):
    require(condition_key(row) == tuple(expected_key), "resumed condition index/configuration mismatch")
    require(row["tokenizer"] in COHORT and row["vocab_budget"] in VOCABS, "invalid primary condition")
    require(type(row["seed"]) is int, "invalid row seed")
    require(
        row.get("git_commit") == identity["commit_hash"]
        and row.get("extension_hash") == identity["extension_hash"],
        "resumed condition provenance mismatch",
    )
    require(row.get("dataset_hash") == dataset["assignment_hash"], "resumed corpus assignment mismatch")
    require(row.get("special_tokens") == SPECIAL_IDS, "resumed special accounting mismatch")
    require(
        row.get("actual_vocab_size") == row["vocab_budget"]
        and row.get("tokenizer_config") == tokenizer_configuration(row["tokenizer"], row["vocab_budget"]),
        "resumed tokenizer/vocabulary configuration mismatch",
    )
    require(row.get("model_kind") == "tokenizer_only" and row.get("model_config") is None, "invalid Phase A model")
    require(row.get("artifact_hashes") and row.get("artifact"), "missing tokenizer artifacts")
    require(row["tokenizer"] != "uniq_superbpe" or row.get("learned_merges", 0) > 0, "SuperBPE zero-merge condition")
    require(
        row.get("training_bytes") == byte_totals(dataset["document_bytes"]["train"]),
        "tokenizer training byte accounting mismatch",
    )
    require(
        row.get("training_input") == tokenizer_training_input(row["tokenizer"]),
        "tokenizer training input representation mismatch",
    )
    require(
        type(row.get("training_wall_clock_seconds")) is float and row["training_wall_clock_seconds"] > 0.0,
        "missing tokenizer training time",
    )
    expected_mb_s = dataset["splits"]["train"][BYTE_BUDGET_FIELD] / (
        row["training_wall_clock_seconds"] * 1_000_000
    )
    require(row.get("training_normalized_mb_per_sec") == expected_mb_s, "invalid tokenizer training throughput")
    for split in ("validation", "test"):
        metric = row[split]
        require(
            metric["utf8_bytes"] == dataset["splits"][split]["utf8_bytes"] and metric["tokens"] > 0,
            "incomplete tokenizer evaluation",
        )
        require(
            all(metric.get(k) == dataset["splits"][split][k] for k in (BYTE_BUDGET_FIELD, BYTE_AUDIT_FIELD)),
            "tokenizer source/normalized byte mismatch",
        )
        require(metric["unicode_characters"] == dataset["splits"][split]["unicode_characters"], "character denominator mismatch")
        require(
            metric["tokens_per_unicode_character"] == metric["tokens"] / metric["unicode_characters"]
            and metric["bytes_per_token"] == metric["utf8_bytes"] / metric["tokens"],
            "invalid tokenizer metric",
        )
        require(
            type(metric.get("byte_fallback_tokens")) is int
            and 0 <= metric["byte_fallback_tokens"] <= metric["tokens"],
            "invalid fallback count",
        )
        require(
            metric.get("byte_fallback_percent") == 100.0 * metric["byte_fallback_tokens"] / metric["tokens"],
            "invalid fallback percentage",
        )
    json.dumps(row, allow_nan=False)
    if artifact_root is not None:
        artifact_directory = Path(artifact_root) / row["artifact"]
        require(
            artifact_directory.resolve().is_relative_to(Path(artifact_root).resolve())
            and artifact_hashes(artifact_directory) == row["artifact_hashes"],
            "resumed tokenizer artifact mismatch",
        )
        load_tokenizer(row, Path(artifact_root))
    return row


def load_phase_a_resume_records(output, meta, identity, dataset):
    require(not (output / "ledger.json").exists(), "Phase A already has a complete ledger")
    require(
        digest(read_json(output / "plan.json")) == digest({**meta, "status": "planned"}),
        "resume plan/provenance mismatch",
    )
    expected = [tuple(condition) for condition in meta["expected_conditions"]]
    files = sorted(output.glob("condition-*.json"))
    require(all(re.fullmatch(r"condition-\d{3}\.json", path.name) for path in files), "invalid condition filename")
    records = {}
    for path in files:
        index = int(path.stem.removeprefix("condition-"))
        require(index < len(expected) and index not in records, "unexpected/duplicate resumed condition")
        payload = read_json(path)
        require(
            set(payload) == {"status", "record"} and payload["status"] == "condition_complete",
            "invalid or incomplete condition envelope",
        )
        records[index] = validate_phase_a_condition_record(payload["record"], identity, dataset, expected[index], output)
    return records


def validate_research_ledger(payload, identity, dataset):
    validate_ledger(payload, expected_commit=identity["commit_hash"])
    meta = payload["metadata"]
    require(meta.get("research_schema_version") == RESEARCH_SCHEMA, "stale research ledger schema")
    require(
        meta.get("byte_budget_field") == BYTE_BUDGET_FIELD and meta.get("byte_audit_field") == BYTE_AUDIT_FIELD,
        "byte budget/audit definition mismatch",
    )
    require(
        meta.get("identity") == identity and meta.get("dataset") == dataset,
        "stale source, extension, environment, or dataset",
    )
    phase = meta.get("phase")
    require(phase in ("A", "B", "C") and meta.get("status") == "complete", "incomplete phase")
    require(meta.get("special_tokens") == SPECIAL_IDS and meta.get("byte_tokens") == 256, "special accounting mismatch")
    rows = payload["records"]
    expected = [tuple(c) for c in meta.get("expected_conditions", [])]
    keys = [condition_key(r) for r in rows]
    require(expected and len(keys) == len(set(keys)) and set(keys) == set(expected), "incomplete/duplicate conditions")
    seeds = meta.get("seeds", [])
    require(len(set(seeds)) == len(seeds) and all(type(s) is int and s >= 0 for s in seeds), "invalid seeds")
    require(len(seeds) == (3 if phase == "C" else 1), "wrong seed count for phase")
    if phase in ("A", "B"):
        regimes = ("tokenizer_only",) if phase == "A" else REGIMES
        grid = {(n, v, r, s) for n in COHORT for v in VOCABS for r in regimes for s in seeds}
        require(set(keys) == grid, "primary cohort/grid incomplete")
    else:
        selected = {tuple(c) for c in meta.get("selected_conditions", [])}
        require(set(keys) == {(*c, s) for c in selected for s in seeds}, "confirmatory selection incomplete")
        require(isinstance(meta.get("selection_hash"), str), "missing preregistered selection")
    for row in rows:
        require(row["tokenizer"] in COHORT and row["vocab_budget"] in VOCABS, "invalid primary condition")
        require(type(row["seed"]) is int and row["seed"] in seeds, "invalid row seed")
        require(row.get("dataset_hash") == dataset["assignment_hash"], "corpus assignment mismatch")
        require(
            row.get("git_commit") == identity["commit_hash"]
            and row.get("extension_hash") == identity["extension_hash"],
            "row provenance mismatch",
        )
        require(row.get("special_tokens") == SPECIAL_IDS, "row special accounting mismatch")
        if phase == "A":
            validate_phase_a_condition_record(row, identity, dataset, condition_key(row))
        else:
            require(
                row["model_kind"] == "causal_transformer" and row["budget_regime"] in REGIMES,
                "invalid causal LM condition",
            )
            require(row.get("model_config") == meta.get("model_config"), "model configuration mismatch")
            require(row.get("requested_budget") == meta["budgets"][row["budget_regime"]], "budget mismatch")
            config = row["model_config"]
            require(
                config == model_config(phase, config["device"]), "architecture or evaluation specification mismatch"
            )
            cfg = LMArchConfig(**config["architecture"])
            require(
                all(
                    type(row.get(k)) is int and row[k] == v
                    for k, v in parameter_accounting(row["vocab_budget"], cfg, config["context"]).items()
                ),
                "incorrect parameter component count",
            )
            artifact_key = f"{row['tokenizer']}:{row['vocab_budget']}"
            require(
                row["tokenizer_artifact_hash"] == meta["tokenizer_artifacts"][artifact_key],
                "tokenizer artifact mismatch",
            )
            require(row["training_steps"] > 0 and row["training_target_tokens"] > 0, "no-op LM condition")
            targets, squares = row["training_target_tokens"], row.get("training_sequence_length_squared_sum")
            require(
                type(targets) is int and type(squares) is int and targets <= squares <= targets * config["context"],
                "invalid FLOP sequence accounting",
            )
            require(
                all(row.get(k) == v for k, v in flop_accounting(row["vocab_budget"], cfg, targets, squares).items()),
                "FLOP component/estimator mismatch",
            )
            require(row.get("training_byte_scope") == "complete_document_prefix", "ambiguous training byte scope")
            completed = row.get("completed_training_documents")
            require(type(completed) is int and completed >= 0, "invalid completed document count")
            expected_bytes = byte_totals(dataset["document_bytes"]["train"], completed)
            require(
                row.get("training_bytes") == expected_bytes
                and row["completed_document_bytes"] == expected_bytes[BYTE_BUDGET_FIELD],
                "training source/normalized byte mismatch",
            )
            if row["budget_regime"] == "bytes":
                require(
                    type(row["requested_budget"]) is int
                    and expected_bytes[BYTE_BUDGET_FIELD] == row["requested_budget"],
                    "normalized byte budget mismatch",
                )
            else:
                require(
                    0.99 * row["requested_budget"] <= row["actual_analytical_flops"] <= row["requested_budget"],
                    "FLOP budget mismatch",
                )
            require(phase != "B" or "test" not in row, "screening must not inspect LM test loss")
            for split in ("validation", "test") if phase == "C" else ("validation",):
                metric = row[split]
                require(
                    metric["target_tokens_including_eos"] == meta["evaluation_targets"][artifact_key][split],
                    "held-out targets missing or duplicated",
                )
                require(
                    metric["utf8_bytes"] == dataset["splits"][split]["utf8_bytes"], "held-out byte denominator mismatch"
                )
                recomputed = nll_metrics(
                    metric["total_nll_nats"],
                    metric["target_tokens_including_eos"],
                    metric["utf8_bytes"],
                    source_utf8_bytes=dataset["splits"][split][BYTE_AUDIT_FIELD],
                )
                require(metric == recomputed, "invalid held-out NLL/BPB/perplexity")
    # Reject nested NaN/Infinity as well as the shared schema's top-level checks.
    json.dumps(payload, allow_nan=False)
    return payload


def load_phase(path, identity, dataset, phase):
    data = validate_research_ledger(read_json(path), identity, dataset)
    require(data["metadata"]["phase"] == phase, f"expected Phase {phase} ledger")
    return data


def select_conditions(selection_path, screening, screening_path):
    selection = read_json(selection_path)
    require(selection.get("schema_version") == 1, "invalid selection schema")
    require(selection.get("screening_sha256") == file_hash(screening_path), "stale screening selection")
    require(
        isinstance(selection.get("rationale"), str) and selection["rationale"].strip(),
        "selection requires validation-based rationale",
    )
    selected = [tuple(c) for c in selection.get("conditions", [])]
    available = {condition_key(r)[:3] for r in screening["records"]}
    require(
        selected and len(set(selected)) == len(selected) and set(selected) <= available,
        "invalid/duplicate/un-screened selection",
    )
    return selected, digest(selection)


def run(args):
    resume = bool(getattr(args, "resume", False))
    require(not resume or args.phase == "A", "resume is supported only for Phase A")
    docs, dataset = load_dataset(args.dataset)
    identity = runtime_identity()
    require(not identity["working_tree_dirty"], "commit the reviewed harness before research runs")
    require(args.phase == "A" or args.phase_a, "LM phases require completed Phase A")
    require(
        args.phase != "A"
        or all(v is None for v in (args.phase_a, args.screening, args.selection, args.flops, args.bytes)),
        "Phase A does not accept LM inputs",
    )
    require(args.phase == "C" or not (args.screening or args.selection), "selection inputs are Phase C only")
    seeds = args.seeds if args.seeds is not None else ([0] if args.phase != "C" else [1, 2, 3])
    require(
        len(seeds) == (3 if args.phase == "C" else 1) and len(set(seeds)) == len(seeds) and all(s >= 0 for s in seeds),
        "phase seed count mismatch",
    )
    meta = {
        **identity,
        "identity": identity,
        "research_schema_version": RESEARCH_SCHEMA,
        "byte_budget_field": BYTE_BUDGET_FIELD,
        "byte_audit_field": BYTE_AUDIT_FIELD,
        "phase": args.phase,
        "status": "complete",
        "dataset": dataset,
        "data_split": "document_disjoint_train_validation_test",
        "special_tokens": SPECIAL_IDS,
        "byte_tokens": 256,
        "seeds": seeds,
    }
    phase_a = None
    if args.phase != "A":
        require(
            args.flops is not None and args.bytes is not None and math.isfinite(args.flops) and args.flops > 0,
            "both positive matching budgets required",
        )
        byte_schedule(docs["train"], args.bytes)
        phase_a = load_phase(args.phase_a, identity, dataset, "A")
        meta["phase_a_sha256"] = file_hash(args.phase_a)
        for row in phase_a["records"]:
            load_tokenizer(row, Path(args.phase_a).parent)
        cfg, context = (SCREEN, 128) if args.phase == "B" else (CONFIRM, 1024)
        if args.phase == "B":
            require(args.flops <= 1e11 and args.bytes <= 1_000_000, "Phase B exceeds small-screening safety caps")
        meta["model_config"] = model_config(args.phase, args.device)
        meta["evaluation_targets"] = {
            f"{r['tokenizer']}:{r['vocab_budget']}": {
                s: r[s]["tokens"] + dataset["splits"][s]["documents"] for s in ("validation", "test")
            }
            for r in phase_a["records"]
        }
        meta["tokenizer_artifacts"] = {
            f"{r['tokenizer']}:{r['vocab_budget']}": digest(r["artifact_hashes"]) for r in phase_a["records"]
        }
        meta["budgets"] = {"flops": args.flops, "bytes": args.bytes}
    selected = (
        [(n, v, "tokenizer_only") for n in COHORT for v in VOCABS]
        if args.phase == "A"
        else [(n, v, r) for n in COHORT for v in VOCABS for r in REGIMES]
    )
    if args.phase == "C":
        require(args.screening and args.selection, "Phase C requires completed screening and explicit selection")
        screening = load_phase(args.screening, identity, dataset, "B")
        require(
            screening["metadata"]["phase_a_sha256"] == meta["phase_a_sha256"],
            "screening used different tokenizer artifacts",
        )
        require(
            all(screening["metadata"][k] == meta[k] for k in ("evaluation_targets", "tokenizer_artifacts")),
            "screening disagrees with Phase A",
        )
        require(set(seeds).isdisjoint(screening["metadata"]["seeds"]), "confirmatory seeds must be new")
        selected, meta["selection_hash"] = select_conditions(args.selection, screening, args.screening)
        meta["selected_conditions"] = selected
        meta["screening_sha256"] = file_hash(args.screening)
    meta["expected_conditions"] = [(*c, s) for c in selected for s in seeds]
    output = Path(args.output)
    if output.exists():
        require(resume and output.is_dir(), "output already exists; use --resume only for an incomplete Phase A")
        existing = load_phase_a_resume_records(output, meta, identity, dataset)
    else:
        output.mkdir(parents=True)
        # If execution fails, this plan and completed artifacts remain; no complete ledger is emitted.
        write_new_json_atomic(output / "plan.json", {**meta, "status": "planned"})
        existing = {}
    rows = []
    for index, (name, vocab, regime, seed) in enumerate(meta["expected_conditions"]):
        if index in existing:
            print(f"Phase A: validated and skipped condition {index:03d} {name} V={vocab}", flush=True)
            rows.append(existing[index])
            continue
        row = {
            "tokenizer": name,
            "vocab_budget": vocab,
            "actual_vocab_size": vocab,
            "seed": seed,
            "budget_regime": regime,
            "dataset_hash": dataset["assignment_hash"],
            "git_commit": identity["commit_hash"],
            "extension_hash": identity["extension_hash"],
            "special_tokens": SPECIAL_IDS,
        }
        print(f"Phase {args.phase}: {name} V={vocab} {regime} seed={seed}", flush=True)
        if args.phase == "A":
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            artifact = f"{name}-{vocab}"
            tok, training_seconds = train_tokenizer(name, docs["train"], vocab, output / artifact)
            validate_tokenizer(tok, vocab)
            row.update(
                actual_vocab_size=len(tok.vocab),
                model_kind="tokenizer_only",
                model_config=None,
                tokenizer_config=tokenizer_configuration(name, vocab),
                artifact=artifact,
                artifact_hashes=artifact_hashes(output / artifact),
                learned_merges=tok.merges,
                training_input=tokenizer_training_input(name),
                training_bytes=byte_totals(dataset["document_bytes"]["train"]),
                training_wall_clock_seconds=training_seconds,
                training_normalized_mb_per_sec=dataset["splits"]["train"][BYTE_BUDGET_FIELD]
                / (training_seconds * 1_000_000),
            )
            row.update(
                {
                    s: token_metrics(tok, docs[s], source_utf8_bytes=dataset["splits"][s][BYTE_AUDIT_FIELD])
                    for s in ("validation", "test")
                }
            )
        else:
            source = next(r for r in phase_a["records"] if (r["tokenizer"], r["vocab_budget"]) == (name, vocab))
            tok = load_tokenizer(source, Path(args.phase_a).parent)
            row.update(
                model_kind="causal_transformer",
                model_config=meta["model_config"],
                requested_budget=meta["budgets"][regime],
                tokenizer_artifact_hash=digest(source["artifact_hashes"]),
            )
            row.update(
                train_lm(
                    tok,
                    docs,
                    cfg,
                    context,
                    regime,
                    meta["budgets"][regime],
                    seed,
                    args.device,
                    args.phase == "C",
                    document_bytes=dataset["document_bytes"],
                )
            )
        write_new_json_atomic(output / f"condition-{index:03d}.json", {"status": "condition_complete", "record": row})
        rows.append(row)
    require(runtime_identity() == identity, "source or environment changed during run")
    payload = {"metadata": meta, "records": rows}
    validate_research_ledger(payload, identity, dataset)
    write_new_json_atomic(output / "ledger.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("A", "B", "C"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-a", type=Path)
    parser.add_argument("--screening", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--flops", type=float)
    parser.add_argument("--bytes", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true", help="Resume a provenance-matched incomplete Phase A directory.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
