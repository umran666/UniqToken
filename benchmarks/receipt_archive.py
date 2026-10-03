"""Store receipted benchmark evidence without expanding generated data in PR diffs."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from benchmarks import run_research_experiments as h


def inspect(archive):
    names = archive.namelist()
    h.require(len(names) == len(set(names)), "duplicate archive entries")
    for name in names:
        path = PurePosixPath(name)
        h.require(
            not path.is_absolute()
            and ".." not in path.parts
            and path.as_posix() == name
            and name not in ("", ".")
            and "\\" not in name
            and ":" not in name,
            "archive path escape",
        )
    h.require("manifest.json" in names, "archive missing manifest")
    receipt = json.loads(archive.read("manifest.json"))
    h.require(
        receipt["status"] == "complete" and "manifest.json" not in receipt["artifacts"],
        "incomplete/self-referential receipt",
    )
    h.require(set(names) == set(receipt["artifacts"]) | {"manifest.json"}, "archive/receipt coverage mismatch")
    for relative, digest in receipt["artifacts"].items():
        h.require(hashlib.sha256(archive.read(relative)).hexdigest() == digest, f"archive receipt mismatch: {relative}")
    return receipt


def pack(root, output):
    h.require(
        not output.exists() and not output.resolve().is_relative_to(root.resolve()),
        "archive must be new and outside source",
    )
    receipt = h.read_json(root / "manifest.json")
    actual = h.artifact_hashes(root)
    actual.pop("manifest.json")
    h.require(actual == receipt["artifacts"], "source receipt coverage mismatch")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(set(receipt["artifacts"]) | {"manifest.json"}):
            source = (root / relative).resolve()
            h.require(source.is_relative_to(root.resolve()), "source receipt path escape")
            data = source.read_bytes()
            if relative != "manifest.json":
                h.require(
                    hashlib.sha256(data).hexdigest() == receipt["artifacts"][relative],
                    "source changed before archiving",
                )
            info = zipfile.ZipInfo(relative, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=9)
    with zipfile.ZipFile(output) as archive:
        inspect(archive)
    return {"archive_sha256": h.file_hash(output), "original_manifest_sha256": h.file_hash(root / "manifest.json")}


def unpack(path, output):
    h.require(not output.exists(), "unpack output must be new")
    with zipfile.ZipFile(path) as archive:
        receipt = inspect(archive)
        output.mkdir(parents=True)
        archive.extractall(output)
    for relative, digest in receipt["artifacts"].items():
        h.require(h.file_hash(output / relative) == digest, "extracted receipt mismatch")
    return output


def publish(root, output):
    h.require(
        not output.exists() and not output.resolve().is_relative_to(root.resolve()),
        "publication must be new and outside source",
    )
    output.mkdir(parents=True)
    binding = pack(root, output / "evidence.zip")
    for relative in h.read_json(root / "manifest.json")["artifacts"]:
        if Path(relative).suffix.lower() in (".md", ".csv", ".png"):
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
    h.write_new_json(
        output / "archive.json",
        {"schema_version": 1, "status": "complete", **binding, "artifacts": h.artifact_hashes(output)},
    )


@contextmanager
def retained_bundle(root):
    """Verify the archive and readable previews before exposing original paths."""
    binding = h.read_json(root / "archive.json")
    h.require(binding["schema_version"] == 1 and binding["status"] == "complete", "unsupported archive binding")
    actual = h.artifact_hashes(root)
    actual.pop("archive.json")
    h.require(actual == binding["artifacts"], "publication receipt mismatch")
    h.require(h.file_hash(root / "evidence.zip") == binding["archive_sha256"], "archive binding mismatch")
    with tempfile.TemporaryDirectory(prefix="uniqtoken-evidence-") as directory:
        extracted = unpack(root / "evidence.zip", Path(directory) / "original")
        h.require(
            h.file_hash(extracted / "manifest.json") == binding["original_manifest_sha256"],
            "original manifest mismatch",
        )
        for relative in binding["artifacts"]:
            if relative != "evidence.zip":
                h.require(
                    h.file_hash(root / relative) == h.file_hash(extracted / relative), "preview differs from original"
                )
        yield extracted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("publish", "unpack"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "publish":
        publish(args.source, args.output)
    else:
        with retained_bundle(args.source) as root:
            h.require(not args.output.exists(), "unpack output must be new")
            shutil.copytree(root, args.output)


if __name__ == "__main__":
    main()
