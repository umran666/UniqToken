"""Preserve exact native worker receipts in small, verified source archives."""

import argparse
from pathlib import Path
import shutil

from benchmarks import receipt_archive as a
from benchmarks import run_research_experiments as h
from benchmarks.receipt_archive import inspect as inspect, unpack as unpack


def pack(root, output):
    binding = a.pack(root, output)
    return {
        "archive_sha256": binding["archive_sha256"],
        "measurement_manifest_sha256": binding["original_manifest_sha256"],
    }


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
