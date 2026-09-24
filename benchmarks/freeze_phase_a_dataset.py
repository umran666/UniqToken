"""Freeze the pinned MADLAD/The Stack/FLORES Phase A corpus into local JSONL files.

This program intentionally has no ``main``/branch/revision defaults. Callers must
pass immutable commit hashes. It writes no final manifest unless all byte quotas,
source hashes, exact deduplication, and train/evaluation near-duplicate checks pass.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url
import numpy as np

from benchmarks.run_research_experiments import NORMALIZATION, RESERVED_CORPUS_TEXT, file_hash, normalize

MB = 1_000_000
MADLAD = "allenai/MADLAD-400"
STACK = "bigcode/the-stack"
STACK_RELEASE = "v1.3"
MADLAD_LICENSE = "ODC-By-1.0"
STACK_LICENSE = "permissive SPDX license recorded per source file"
FLORES_LICENSE = "record the exact upstream/mirror license in --flores-license"
NEAR_METHOD = "character_13gram_minhash16_lsh4_jaccard_0.85"
MINHASH_HASH_FAMILY = "blake2b64_affine_uint64_v1"
UINT64_MASK = (1 << 64) - 1
MINHASH_COEFFICIENTS = tuple(
    (
        ((0x9E3779B185EBCA87 + seed * 0x517CC1B727220A95) & UINT64_MASK) | 1,
        (0x6A09E667F3BCC909 + seed * 0x94D049BB133111EB) & UINT64_MASK,
    )
    for seed in range(16)
)
STACK_LICENSE_COLUMNS = (
    "max_stars_repo_licenses",
    "max_issues_repo_licenses",
    "max_forks_repo_licenses",
)

# The supplied strata are exact normalized-byte quotas. Equal language shares are
# deterministic and stated in the manifest rather than inferred after sampling.
PROSE_STRATA = {
    "latin_english": (160 * MB, ("en",)),
    "indic_cjk_arabic": (
        160 * MB,
        ("hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "zh", "ja", "ko", "ar", "fa", "ur"),
    ),
    "cyrillic_african": (80 * MB, ("ru", "uk", "bg", "sw", "yo", "am")),
}
CODE_SOURCE_DIRECTORIES = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "sql": "sql",
    "c": "c",
    "cpp": "c++",
    "rust": "rust",
    "go": "go",
}
CODE_LANGUAGES = tuple(CODE_SOURCE_DIRECTORIES)
FLORES_LANGUAGE_FILES = {
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "fa": "pes_Arab",
    "ur": "urd_Arab",
    "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl",
    "bg": "bul_Cyrl",
    "sw": "swh_Latn",
    "yo": "yor_Latn",
    "am": "amh_Ethi",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixed_revision(value: str) -> str:
    require(re.fullmatch(r"[0-9a-f]{40}", value) is not None, "use an immutable 40-character dataset commit hash")
    return value


def quota_by_language(total: int, languages: tuple[str, ...]) -> dict[str, int]:
    base, remainder = divmod(total, len(languages))
    return {language: base + int(index < remainder) for index, language in enumerate(languages)}


def source_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def json_line(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def copy_source(
    repo: str,
    revision: str,
    remote_path: str,
    work: Path,
    license_name: str,
    release_variant: str,
    token: str | None,
) -> dict[str, Any]:
    """Download one immutable source file into the future manifest directory."""
    local_dir = work / "sources" / repo.replace("/", "--") / revision
    existing = local_dir / remote_path
    downloaded = (
        existing
        if existing.is_file()
        else Path(
            hf_hub_download(
                repo,
                remote_path,
                repo_type="dataset",
                revision=revision,
                local_dir=local_dir,
                token=token,
            )
        )
    )
    relative = downloaded.relative_to(work).as_posix()
    return {
        "local_path": relative,
        "sha256": file_hash(downloaded),
        "dataset": repo,
        "revision": revision,
        "url": source_url(repo, revision, remote_path),
        "license": license_name,
        "release_variant": release_variant,
        "remote_path": remote_path,
        "file_bytes": downloaded.stat().st_size,
    }


def repo_files(api: HfApi, repo: str, revision: str, prefix: str, *, filename_prefix: str | None = None) -> list[str]:
    paths = [
        entry.path
        for entry in api.list_repo_tree(
            repo, repo_type="dataset", revision=revision, path_in_repo=prefix, recursive=False
        )
        if hasattr(entry, "size")
        and getattr(entry, "size")
        and (filename_prefix is None or Path(entry.path).name.startswith(filename_prefix))
    ]
    require(paths, f"no files at pinned {repo}@{revision}:{prefix}")
    return sorted(paths, key=lambda path: hashlib.sha256(path.encode()).hexdigest())


def preflight_source_access(repo: str, revision: str, remote_path: str, token: str | None) -> None:
    """Confirm a pinned data object is downloadable without transferring its body."""
    metadata = get_hf_file_metadata(
        hf_hub_url(repo, remote_path, repo_type="dataset", revision=revision), token=token, timeout=30
    )
    require(metadata.size is not None and metadata.size > 0, f"empty or inaccessible source: {repo}:{remote_path}")


def source_record(
    text: str,
    *,
    identifier: str,
    language: str,
    domain: str,
    source: dict[str, Any],
    source_record: int,
    truncated: bool,
) -> dict[str, Any]:
    normalized = normalize(text)
    require(normalized and normalized.strip(), "empty selected normalized document")
    return {
        "id": identifier,
        "text": text,
        "language": language,
        "domain": domain,
        "raw_utf8_bytes": len(text.encode("utf-8")),
        "normalized_utf8_bytes": len(normalized.encode("utf-8")),
        "source": {
            **{key: value for key, value in source.items() if key != "local_path_abs"},
            "source_file_sha256": source["sha256"],
            "source_record": source_record,
        },
        "dedup": {
            "status": "accepted_after_exact_and_near_eval_check",
            "method": NEAR_METHOD,
            "truncated_to_quota": truncated,
        },
    }


def normalized_prefix(text: str, maximum: int) -> str | None:
    """A UTF-8-safe raw prefix whose whole-string normalization fits exactly."""
    if maximum <= 0 or not text:
        return None

    sizes: dict[int, int] = {}

    def normalized_size(end: int) -> int:
        if end not in sizes:
            sizes[end] = len(normalize(text[:end]).encode("utf-8"))
        return sizes[end]

    # NFKC byte length is not formally monotone at every raw code-point
    # boundary. A binary search therefore supplies candidates only: every
    # returned prefix is verified exactly, and an ambiguous record is skipped.
    # The local probe covers ordinary UTF-8 width jumps and combining sequences
    # without repeatedly normalizing every prefix of a large source document.
    low, high = 1, len(text)
    while low <= high:
        middle = (low + high) // 2
        size = normalized_size(middle)
        if size < maximum:
            low = middle + 1
        elif size > maximum:
            high = middle - 1
        else:
            return text[:middle]

    for end in range(max(1, low - 64), min(len(text), low + 64) + 1):
        if normalized_size(end) == maximum:
            return text[:end]
    return None


def append_exact(
    records: Iterable[tuple[str, dict[str, Any], int]],
    target: int,
    output: Path,
    *,
    language: str,
    domain: str,
    dedup_index: DedupIndex | None = None,
) -> dict[str, int]:
    """Write a quota exactly, only slicing a final source document on a character boundary."""
    used, raw_used, documents = 0, 0, 0
    with output.open("wb") as stream:
        for text, source, record_index in records:
            if not isinstance(text, str) or not text.strip():
                continue
            if RESERVED_CORPUS_TEXT.search(text):
                continue
            normalized_size = len(normalize(text).encode("utf-8"))
            remaining = target - used
            if normalized_size <= remaining:
                chosen, truncated = text, False
            else:
                chosen = normalized_prefix(text, remaining)
                if chosen is None:
                    continue
                truncated = True
            if dedup_index is not None and not dedup_index.accept(chosen):
                continue
            record = source_record(
                chosen,
                identifier=f"{source['dataset']}:{source['remote_path']}:{record_index}:{used}",
                language=language,
                domain=domain,
                source=source,
                source_record=record_index,
                truncated=truncated,
            )
            stream.write(json_line(record))
            used += record["normalized_utf8_bytes"]
            raw_used += record["raw_utf8_bytes"]
            documents += 1
            if used == target:
                return {
                    "documents": documents,
                    "raw_utf8_bytes": raw_used,
                    "normalized_utf8_bytes": used,
                }
    raise ValueError(f"could not meet exact {target} normalized-byte quota for {language}/{domain}")


def completed_part(path: Path, target: int, *, language: str, domain: str) -> bool:
    """Validate a resumable partition; incomplete partitions may be regenerated."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    total = 0
    for row in records_from_jsonl(path):
        require(row.get("language") == language and row.get("domain") == domain, "resumed partition metadata mismatch")
        text = row.get("text")
        require(isinstance(text, str), "resumed partition contains invalid text")
        if RESERVED_CORPUS_TEXT.search(text):
            return False
        require(
            row.get("raw_utf8_bytes") == len(text.encode("utf-8"))
            and row.get("normalized_utf8_bytes") == len(normalize(text).encode("utf-8")),
            "resumed partition byte accounting mismatch",
        )
        total += row["normalized_utf8_bytes"]
    require(total <= target, "resumed partition exceeds its normalized-byte quota")
    return total == target


def resume_sources(path: Path, work: Path) -> list[dict[str, Any]]:
    """Recover and verify the source inventory referenced by a completed partition."""
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records_from_jsonl(path):
        recorded = row["source"]
        key = (recorded["dataset"], recorded["remote_path"])
        if key in sources:
            continue
        local = work / recorded["local_path"]
        require(local.is_file(), "resumed source file is missing")
        require(file_hash(local) == recorded["source_file_sha256"], "resumed source hash mismatch")
        sources[key] = {
            "local_path": recorded["local_path"],
            "local_path_abs": str(local),
            "sha256": recorded["source_file_sha256"],
            "dataset": recorded["dataset"],
            "revision": recorded["revision"],
            "url": recorded["url"],
            "license": STACK_LICENSE if recorded["dataset"] == STACK else recorded["license"],
            "release_variant": recorded["release_variant"],
            "remote_path": recorded["remote_path"],
            "file_bytes": local.stat().st_size,
        }
    return list(sources.values())


def madlad_records(source: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any], int]]:
    with gzip.open(Path(source["local_path_abs"]), "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            text = row.get("text")
            if isinstance(text, str):
                yield text, source, index


def stack_license_allowlist(source: dict[str, Any]) -> frozenset[str]:
    values = json.loads(Path(source["local_path_abs"]).read_text(encoding="utf-8"))
    require(isinstance(values, list) and values, "The Stack pinned license allowlist is empty")
    normalized = frozenset(str(value).strip().lower() for value in values if str(value).strip())
    require(len(normalized) == len(values), "The Stack pinned license allowlist contains duplicates or invalid IDs")
    return normalized


def stack_records(
    source: dict[str, Any], permissive_licenses: frozenset[str]
) -> Iterable[tuple[str, dict[str, Any], int]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(source["local_path_abs"])
    require(set(STACK_LICENSE_COLUMNS) <= set(parquet.schema_arrow.names), "The Stack license columns are missing")
    index = 0
    for batch in parquet.iter_batches(batch_size=1024, columns=["content", *STACK_LICENSE_COLUMNS], use_threads=False):
        for row in batch.to_pylist():
            text = row["content"]
            licenses = {
                str(value).strip().lower()
                for column in STACK_LICENSE_COLUMNS
                for value in (row[column] if isinstance(row[column], list) else [row[column]])
                if value is not None and str(value).strip()
            }
            if isinstance(text, str) and licenses and licenses <= permissive_licenses:
                yield text, {**source, "license": ",".join(sorted(licenses))}, index
            index += 1


def flores_records(source: dict[str, Any]) -> Iterable[tuple[str, str, int]]:
    """Yield language, sentence, and row index from the official `all` parquet config."""
    import pyarrow.parquet as pq

    columns = ["id", *(f"sentence_{code}" for code in FLORES_LANGUAGE_FILES.values())]
    table = pq.read_table(source["local_path_abs"], columns=columns)
    for row_index, row in enumerate(table.to_pylist()):
        source_index = row.get("id", row_index)
        for language, code in FLORES_LANGUAGE_FILES.items():
            text = row.get(f"sentence_{code}")
            require(isinstance(text, str) and text.strip(), f"FLORES missing {code} sentence")
            yield language, text, source_index


def minhash_signature(text: str) -> tuple[int, ...]:
    grams = {text[index : index + 13] for index in range(max(0, len(text) - 12))} or {text}
    base_hashes = np.fromiter(
        (int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big") for gram in grams),
        dtype=np.uint64,
        count=len(grams),
    )
    return tuple(
        int((base_hashes * np.uint64(multiplier) + np.uint64(offset)).min())
        for multiplier, offset in MINHASH_COEFFICIENTS
    )


def near_duplicate(a: str, b: str) -> bool:
    a_grams = {a[index : index + 13] for index in range(max(0, len(a) - 12))} or {a}
    b_grams = {b[index : index + 13] for index in range(max(0, len(b) - 12))} or {b}
    return len(a_grams & b_grams) / len(a_grams | b_grams) >= 0.85


class DedupIndex:
    """Exact and LSH-indexed near-duplicate state shared by all train partitions."""

    def __init__(self) -> None:
        self.hashes: set[str] = set()
        self.buckets: dict[tuple[int, ...], list[str]] = {}

    @staticmethod
    def band_keys(signature: tuple[int, ...]) -> set[tuple[int, ...]]:
        return {signature[band * 4 : (band + 1) * 4] for band in range(4)}

    def add(self, text: str) -> str | None:
        normalized = normalize(text)
        digest = sha256_text(normalized)
        if digest in self.hashes:
            return "exact"
        signature = minhash_signature(normalized)
        keys = self.band_keys(signature)
        if any(near_duplicate(normalized, candidate) for key in keys for candidate in self.buckets.get(key, [])):
            return "near"
        self.hashes.add(digest)
        for key in keys:
            self.buckets.setdefault(key, []).append(normalized)
        return None

    def accept(self, text: str) -> bool:
        return self.add(text) is None

    def overlaps(self, text: str) -> tuple[bool, bool]:
        normalized = normalize(text)
        exact = sha256_text(normalized) in self.hashes
        signature = minhash_signature(normalized)
        near = any(
            near_duplicate(normalized, candidate)
            for key in self.band_keys(signature)
            for candidate in self.buckets.get(key, [])
        )
        return exact, near


def records_from_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def validate_dedup(train: Path, evaluation: list[Path]) -> None:
    """Reject exact/near training duplicates and any train/evaluation overlap."""
    index = DedupIndex()
    for row in records_from_jsonl(train):
        reason = index.add(row["text"])
        require(reason != "exact", "duplicate normalized training document")
        require(reason != "near", "near duplicate training document")
    evaluation_hashes: set[str] = set()
    for path in evaluation:
        for row in records_from_jsonl(path):
            normalized_hash = sha256_text(normalize(row["text"]))
            require(normalized_hash not in evaluation_hashes, "duplicate normalized evaluation document")
            evaluation_hashes.add(normalized_hash)
            exact, near = index.overlaps(row["text"])
            require(not exact, "exact train/evaluation overlap invalidates manifest")
            require(not near, "near train/evaluation overlap invalidates manifest")


def restore_dedup_index(path: Path, index: DedupIndex) -> None:
    """Rebuild cross-partition dedup state from a completed staged partition."""
    for row in records_from_jsonl(path):
        require(index.accept(row["text"]), "resumed partition violates training deduplication")


def selection_groups(splits: dict[str, Path]) -> list[dict[str, Any]]:
    """Aggregate auditable selected-byte totals without discarding language/domain labels."""
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for split, path in splits.items():
        for row in records_from_jsonl(path):
            source = row["source"]
            key = (split, source["dataset"], source["release_variant"], row["language"], row["domain"])
            group = groups.setdefault(
                key,
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
    return [groups[key] for key in sorted(groups)]


def validate_selection(groups: list[dict[str, Any]]) -> None:
    """Enforce the pre-registered Phase A composition before publishing a manifest."""
    require(groups and all(group["documents"] > 0 for group in groups), "empty selected corpus group")
    require(
        all(group["raw_utf8_bytes"] > 0 and group["normalized_utf8_bytes"] > 0 for group in groups),
        "selected corpus group lacks byte accounting",
    )
    train = [group for group in groups if group["split"] == "train"]
    for stratum, (total, languages) in PROSE_STRATA.items():
        selected = [group for group in train if group["domain"] == stratum]
        require({group["language"] for group in selected} == set(languages), f"incomplete {stratum} coverage")
        require(
            sum(group["normalized_utf8_bytes"] for group in selected) == total,
            f"incorrect {stratum} normalized-byte quota",
        )
        require(
            all(
                group["dataset"] == MADLAD and group["release_variant"] == "data-v1p5/clean_docs_v2"
                for group in selected
            ),
            f"incorrect {stratum} source",
        )
    code = [group for group in train if group["domain"] == "code"]
    require({group["language"] for group in code} == set(CODE_LANGUAGES), "incomplete code-language coverage")
    require(sum(group["normalized_utf8_bytes"] for group in code) == 100 * MB, "incorrect code byte quota")
    require(
        all(group["dataset"] == STACK and group["release_variant"] == STACK_RELEASE for group in code),
        "incorrect code source",
    )
    require(
        sum(group["normalized_utf8_bytes"] for group in train if group["domain"] != "code") == 400 * MB,
        "incorrect prose byte quota",
    )
    for split in ("validation", "test"):
        evaluation = [group for group in groups if group["split"] == split]
        require(
            {group["language"] for group in evaluation} == set(FLORES_LANGUAGE_FILES),
            f"incomplete FLORES-200 {split} coverage",
        )
        require(
            all(
                group["domain"] == "flores200" and group["release_variant"] == "FLORES-200/all" for group in evaluation
            ),
            f"incorrect FLORES-200 {split} source",
        )


def build_manifest(
    work: Path, source_files: list[dict[str, Any]], splits: dict[str, Path], *, revisions: dict[str, str]
) -> dict[str, Any]:
    groups = selection_groups(splits)
    validate_selection(groups)
    return {
        "schema_version": 2,
        "dataset_id": "phase-a-madlad400-data-v1p5-clean-docs-v2-the-stack-v1p3-flores200",
        "source": "Pinned local source inventory; no network access is permitted by the experiment runner.",
        "license": "MADLAD ODC-By; The Stack per-record permissive licenses; FLORES license recorded per pinned source.",
        "deduplication": (
            f"Training records are filtered by exact normalized SHA-256 and {NEAR_METHOD} "
            f"using {MINHASH_HASH_FAMILY}; "
            "rejected records are replaced before quota accounting. Any exact or near train/evaluation overlap aborts freezing."
        ),
        "normalization": NORMALIZATION,
        "freeze": {
            "immutable": True,
            "source_files": source_files,
            "source_revisions": revisions,
            "selection": {
                "prose_normalized_utf8_bytes": 400 * MB,
                "code_normalized_utf8_bytes": 100 * MB,
                "byte_unit": "MB_decimal",
                "flores_devtest_role": "test_only",
                "groups": groups,
            },
        },
        "splits": {name: {"path": path.name, "sha256": file_hash(path)} for name, path in splits.items()},
    }


def run(args: argparse.Namespace) -> Path:
    madlad_revision, stack_revision, flores_revision = map(
        fixed_revision, (args.madlad_revision, args.stack_revision, args.flores_revision)
    )
    require(args.stack_release == STACK_RELEASE, f"Phase A requires The Stack release {STACK_RELEASE}")
    output = Path(args.output).resolve()
    require(not output.exists(), "refuse to overwrite a frozen dataset directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    madlad_files = {
        language: repo_files(
            api,
            MADLAD,
            madlad_revision,
            f"data-v1p5/{language}",
            filename_prefix="clean_docs_v2-",
        )
        for _, languages in PROSE_STRATA.values()
        for language in languages
    }
    stack_files = {
        language: repo_files(api, STACK, stack_revision, f"data/{source_directory}")
        for language, source_directory in CODE_SOURCE_DIRECTORIES.items()
    }
    preflight_source_access(MADLAD, madlad_revision, madlad_files["en"][0], token)
    preflight_source_access(STACK, stack_revision, "licenses.json", token)
    preflight_source_access(STACK, stack_revision, stack_files["python"][0], token)
    for folder in ("dev", "devtest"):
        preflight_source_access(
            args.flores_repo,
            flores_revision,
            f"data/all/{folder}-00000-of-00001.parquet",
            token,
        )
    if args.resume_partial is None:
        work = Path(tempfile.mkdtemp(prefix=f"{output.name}.partial-", dir=output.parent))
    else:
        work = Path(args.resume_partial).resolve()
        require(
            work.is_dir()
            and work.parent == output.parent
            and work.name.startswith(f"{output.name}.partial-")
            and not (work / "manifest.json").exists(),
            "resume directory is not a valid unpublished freeze staging directory",
        )
    try:
        source_files: list[dict[str, Any]] = []
        source_by_key: dict[tuple[str, str], dict[str, Any]] = {}

        def register_source(source: dict[str, Any]) -> dict[str, Any]:
            key = (source["dataset"], source["remote_path"])
            previous = source_by_key.get(key)
            require(previous is None or previous["sha256"] == source["sha256"], "conflicting resumed source")
            if previous is None:
                source_by_key[key] = source
                source_files.append({k: v for k, v in source.items() if k != "local_path_abs"})
            return source_by_key[key]

        def get_source(
            repo: str, revision: str, remote_path: str, license_name: str, release_variant: str
        ) -> dict[str, Any]:
            key = (repo, remote_path)
            if key not in source_by_key:
                source = copy_source(repo, revision, remote_path, work, license_name, release_variant, token)
                source["local_path_abs"] = str(work / source["local_path"])
                register_source(source)
            return source_by_key[key]

        stack_license_source = get_source(
            STACK,
            stack_revision,
            "licenses.json",
            STACK_LICENSE,
            args.stack_release,
        )
        permissive_stack_licenses = stack_license_allowlist(stack_license_source)

        train_parts: list[Path] = []
        train_dedup = DedupIndex()
        for stratum, (total, languages) in PROSE_STRATA.items():
            for language, quota in quota_by_language(total, languages).items():
                part = work / f"train-prose-{language}.jsonl"
                files = madlad_files[language]

                if completed_part(part, quota, language=language, domain=stratum):
                    restore_dedup_index(part, train_dedup)
                    for source in resume_sources(part, work):
                        register_source(source)
                    train_parts.append(part)
                    continue

                def records() -> Iterable[tuple[str, dict[str, Any], int]]:
                    for remote in files:
                        yield from madlad_records(
                            get_source(
                                MADLAD,
                                madlad_revision,
                                remote,
                                MADLAD_LICENSE,
                                "data-v1p5/clean_docs_v2",
                            )
                        )

                append_exact(
                    records(),
                    quota,
                    part,
                    language=language,
                    domain=stratum,
                    dedup_index=train_dedup,
                )
                train_parts.append(part)

        code_total = 100 * MB
        for language, quota in quota_by_language(code_total, CODE_LANGUAGES).items():
            part = work / f"train-code-{language}.jsonl"
            files = stack_files[language]

            if completed_part(part, quota, language=language, domain="code"):
                restore_dedup_index(part, train_dedup)
                for source in resume_sources(part, work):
                    register_source(source)
                train_parts.append(part)
                continue

            def records() -> Iterable[tuple[str, dict[str, Any], int]]:
                for remote in files:
                    yield from stack_records(
                        get_source(STACK, stack_revision, remote, STACK_LICENSE, args.stack_release),
                        permissive_stack_licenses,
                    )

            append_exact(
                records(),
                quota,
                part,
                language=language,
                domain="code",
                dedup_index=train_dedup,
            )
            train_parts.append(part)

        train = work / "train.jsonl"
        with train.open("wb") as joined:
            for part in train_parts:
                joined.write(part.read_bytes())

        # FLORES dev is validation; devtest is test-only, never sampled into training.
        splits: dict[str, Path] = {"train": train}
        evaluation_hashes: set[str] = set()
        for split, folder in (("validation", "dev"), ("test", "devtest")):
            target = work / f"{split}.jsonl"
            remote = f"data/all/{folder}-00000-of-00001.parquet"
            source = get_source(
                args.flores_repo,
                flores_revision,
                remote,
                args.flores_license,
                "FLORES-200/all",
            )
            with target.open("wb") as stream:
                for language, text, index in flores_records(source):
                    normalized_hash = sha256_text(normalize(text))
                    if normalized_hash in evaluation_hashes:
                        continue
                    evaluation_hashes.add(normalized_hash)
                    stream.write(
                        json_line(
                            source_record(
                                text,
                                identifier=f"{args.flores_repo}:{remote}:{language}:{index}",
                                language=language,
                                domain="flores200",
                                source=source,
                                source_record=index,
                                truncated=False,
                            )
                        )
                    )
            splits[split] = target

        validate_dedup(train, [splits["validation"], splits["test"]])
        manifest = build_manifest(
            work,
            source_files,
            splits,
            revisions={
                "madlad": madlad_revision,
                "the_stack": stack_revision,
                "the_stack_release": args.stack_release,
                "flores": flores_revision,
            },
        )
        (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(work, output)
        return output / "manifest.json"
    except Exception:
        print(f"freeze staging preserved for validated resume: {work}", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--madlad-revision", required=True)
    parser.add_argument("--stack-revision", required=True)
    parser.add_argument("--stack-release", default=STACK_RELEASE)
    parser.add_argument("--flores-repo", required=True, help="Exact FLORES-200 repository approved for this run.")
    parser.add_argument("--flores-revision", required=True)
    parser.add_argument("--flores-license", required=True)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--resume-partial", type=Path, default=None)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
