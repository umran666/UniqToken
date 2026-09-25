"""Verify the v1 combined wheel in an isolated environment and emit a receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import venv
import zipfile


EXPECTED_DISTRIBUTION = "uniqtoken-core"
EXPECTED_VERSION = "1.0.0"
REQUIRED_PYTHON_FILES = {
    "benchmarks/__init__.py",
    "uniqtoken/__init__.py",
    "uniqtoken/cli.py",
    "uniqtoken/tokenizer.py",
    "uniqtoken/unigram_trainer.py",
    "uniqtoken/cem_merger.py",
    "uniqtoken_core/__init__.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def wheel_contents(wheel: Path) -> tuple[list[str], str, str]:
    with zipfile.ZipFile(wheel) as archive:
        names = sorted(archive.namelist())
        missing = sorted(REQUIRED_PYTHON_FILES.difference(names))
        if missing:
            raise RuntimeError(f"wheel is missing public Python modules: {missing}")
        native = [
            name for name in names if name.startswith("uniqtoken_core") and name.lower().endswith((".pyd", ".so"))
        ]
        if len(native) != 1:
            raise RuntimeError(f"expected one uniqtoken_core native extension, found {native}")
        native_sha256 = hashlib.sha256(archive.read(native[0])).hexdigest()
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise RuntimeError(f"expected one .dist-info/METADATA, found {metadata}")
        text = archive.read(metadata[0]).decode("utf-8")
        if f"Name: {EXPECTED_DISTRIBUTION}" not in text or f"Version: {EXPECTED_VERSION}" not in text:
            raise RuntimeError("wheel metadata does not identify uniqtoken-core 1.0.0")
        requirements = [line for line in text.splitlines() if line.startswith("Requires-Dist:")]
        if any(
            re.match(r"Requires-Dist:\s*uniq[-_]?token(?:\s|\[|\(|;|$)", line, re.IGNORECASE) for line in requirements
        ):
            raise RuntimeError("wheel depends on the unrelated legacy uniqtoken distribution")
    return names, native[0], native_sha256


SMOKE_CODE = r"""
import importlib.metadata
import json
from math import log

import uniqtoken
import uniqtoken_core
from uniqtoken import CustomTokenizer, Normalizer, RegexPreTokenizer, UnigramModel

vocab = {"tok": log(0.5), "en": log(0.3), "ize": log(0.2)}
token_to_id = {token: index for index, token in enumerate(vocab)}
model = UnigramModel(
    vocab=vocab,
    token_to_id=token_to_id,
    id_to_token={index: token for token, index in token_to_id.items()},
    special_tokens=[],
    max_subword_len=3,
    byte_fallback=False,
)
tokenizer = CustomTokenizer(
    normalizer=Normalizer(normalize_unicode=False),
    pre_tokenizer=RegexPreTokenizer(),
    model=model,
)
ids = tokenizer.encode_to_ids("tokenize")
decoded = tokenizer.decode(ids)
assert ids == [0, 1, 2], ids
assert decoded == "tokenize", decoded
assert uniqtoken.__version__ == "1.0.0"
assert importlib.metadata.version("uniqtoken-core") == "1.0.0"
assert hasattr(uniqtoken_core, "RustPrefixTrie")
print(json.dumps({
    "distribution": "uniqtoken-core",
    "version": uniqtoken.__version__,
    "ids": ids,
    "decoded": decoded,
    "native_module": uniqtoken_core.__name__,
}, sort_keys=True))
"""


ABSENT_CODE = r"""
import importlib.metadata
import importlib.util

assert importlib.util.find_spec("uniqtoken") is None
assert importlib.util.find_spec("uniqtoken_core") is None
try:
    importlib.metadata.version("uniqtoken-core")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("uniqtoken-core distribution metadata remains after uninstall")
"""


def isolated_verification(wheel: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="uniqtoken-wheel-verification-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        if os.name == "nt":
            python = environment / "Scripts" / "python.exe"
        else:
            python = environment / "bin" / "python"

        command([str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)])
        first = json.loads(command([str(python), "-I", "-c", SMOKE_CODE], cwd=root))
        cli = python.parent / ("uniqtoken.exe" if os.name == "nt" else "uniqtoken")
        cli_help = command([str(cli), "--help"], cwd=root)
        if "usage" not in cli_help.lower():
            raise RuntimeError("installed uniqtoken CLI did not produce help output")

        command([str(python), "-m", "pip", "uninstall", "-y", EXPECTED_DISTRIBUTION])
        command([str(python), "-I", "-c", ABSENT_CODE], cwd=root)

        command(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                str(wheel),
            ]
        )
        second = json.loads(command([str(python), "-I", "-c", SMOKE_CODE], cwd=root))
        return {
            "python_version": command([str(python), "--version"]),
            "initial_install": first,
            "cli_help_verified": True,
            "uninstall_removed_public_and_native_imports": True,
            "reinstall": second,
        }


def git_value(*args: str) -> str:
    try:
        return command(["git", *args], cwd=Path(__file__).resolve().parents[1])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    wheel = args.wheel.resolve()
    if not wheel.is_file():
        raise FileNotFoundError(wheel)
    files, native_name, native_sha256 = wheel_contents(wheel)
    verification = isolated_verification(wheel)
    receipt = {
        "schema_version": 1,
        "status": "packaging_contract_verified",
        "source_commit": git_value("rev-parse", "HEAD"),
        "working_tree_clean": git_value("status", "--porcelain") == "",
        "builder_python": platform.python_version(),
        "platform": platform.platform(),
        "distribution": EXPECTED_DISTRIBUTION,
        "version": EXPECTED_VERSION,
        "wheel": wheel.name,
        "wheel_sha256": sha256(wheel),
        "wheel_files": files,
        "native_extension": native_name,
        "native_extension_sha256": native_sha256,
        "verification": verification,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "wheel", "wheel_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
