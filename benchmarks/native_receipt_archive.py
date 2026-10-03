"""Preserve exact native worker receipts in small, verified source archives."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from benchmarks import run_research_experiments as h


def inspect(archive):
    names = archive.namelist()
    h.require(len(names) == len(set(names)), "duplicate archive entries")
    for name in names:
        h.require(
            not Path(name).is_absolute() and ".." not in Path(name).parts and "\\" not in name, "archive path escape"
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
    return {"archive_sha256": h.file_hash(output), "measurement_manifest_sha256": h.file_hash(root / "manifest.json")}


def unpack(path, output):
    h.require(not output.exists(), "unpack output must be new")
    with zipfile.ZipFile(path) as archive:
        receipt = inspect(archive)
        output.mkdir(parents=True)
        archive.extractall(output)
    for relative, digest in receipt["artifacts"].items():
        h.require(h.file_hash(output / relative) == digest, "extracted receipt mismatch")
    return output


def publish(args):
    output = args.output
    h.require(not output.exists(), "publication must be new")
    identity = h.runtime_identity()
    h.require(not identity["working_tree_dirty"], "commit archive publisher before export")
    output.mkdir(parents=True)
    sources = {}
    for name in ("baseline", "paired", "prefix"):
        sources[name] = pack(getattr(args, name), output / f"{name}-receipts.zip")
    summary = output / "summary"
    shutil.copytree(args.summary, summary)
    for relative, digest in h.read_json(summary / "manifest.json")["artifacts"].items():
        h.require(h.file_hash(summary / relative) == digest, "summary changed while publishing")
    h.require(h.runtime_identity() == identity, "archive publication source/runtime changed")
    h.write_new_json(
        output / "manifest.json",
        {
            "status": "complete",
            "sources": sources,
            "publication_identity": identity,
            "artifacts": h.artifact_hashes(output),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "paired", "prefix", "summary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    publish(parser.parse_args())


if __name__ == "__main__":
    main()
