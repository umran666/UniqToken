"""Short CPU-only Phase C stack calibration; never publishes an LM condition."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import modal


HERE = Path(__file__).resolve().parent
app = modal.App("uniqtoken-phase-c-cpu-calibration-v1")
authorized = modal.Volume.from_name("uniqtoken-phase-b-authorized-v1")
historical = modal.Volume.from_name("uniqtoken-frozen-2ad27746")
phase_a = modal.Volume.from_name("uniqtoken-screen-results-996536b")
phase_b_results = modal.Volume.from_name("uniqtoken-phase-b-screen-771d62a")
feasibility = modal.Volume.from_name("uniqtoken-phase-c-feasibility-81bea81")
results = modal.Volume.from_name("uniqtoken-phase-c-cpu-calibration-v1", create_if_missing=True)

image = (
    modal.Image.from_registry("python:3.10.17-slim-bookworm")
    .apt_install("git", "curl", "build-essential", "pkg-config")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.90.0")
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin", "PYTHONUNBUFFERED": "1",
          "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0", "RAYON_NUM_THREADS": "1"})
    .pip_install("typing-extensions==4.12.2")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy==2.2.6", "sentencepiece==0.2.1", "regex==2026.4.4", "maturin==1.9.6")
    .add_local_file(HERE / "repo-phase-c-cpu-calibration.bundle", "/tmp/repo.bundle", copy=True)
    .run_commands(
        "git clone /tmp/repo.bundle /opt/uniqtoken",
        "cd /opt/uniqtoken && maturin build --release --locked --manifest-path crates/uniqtoken_core/Cargo.toml -o /tmp/wheels",
        "pip install /tmp/wheels/*.whl",
    )
)


def arguments(output):
    return [
        sys.executable, "-m", "tools.phase_c_gpu_calibration",
        "--phase-a", "/phase_a/screen-migrated-f57b93d/ledger.json",
        "--dataset", "/historical/manifest.json",
        "--selection", "/authorized/selection.json",
        "--selection-sha256", "d3449786b636216057e01e1372b4805c1181c481ca521a74c9e29e07676a0ea4",
        "--source-manifest", "/authorized/source/manifest.json",
        "--exposure-manifest", "/opt/uniqtoken/artifacts/phase-c-exposure-v3/frozen/exposure.json",
        "--exposure-manifest-sha256", "88c6661699025bb90c1a42b9b24adbc3b523b79282f3925e6a40653f7e4fa5f8",
        "--validation-receipt", "/opt/uniqtoken/artifacts/phase-c-exposure-v3/frozen/confirmation-validation.json",
        "--validation-receipt-sha256", "e58d1c961ac4220ae8b0d48278702a09c851956dfdf0a6a3f30250135dbc47c1",
        "--phase-b-ledger", "/phase_b_results/phase-b-screen-771d62a/ledger.json",
        "--phase-b-report", "/opt/uniqtoken/benchmarks/PHASE_B_ANALYSIS_REPORT.md",
        "--protocol", "/opt/uniqtoken/benchmarks/PHASE_C_CONFIRMATORY_PROTOCOL.md",
        "--feasibility-receipt", "/feasibility/phase-c-feasibility-parallel-441770d/receipt.json",
        "--device", "cpu", "--hourly-rate", "0.158256", "--output", str(output),
    ]


@app.function(
    image=image, cpu=2, memory=8192, timeout=900, max_containers=1, retries=0,
    volumes={"/authorized": authorized, "/historical": historical, "/phase_a": phase_a,
             "/phase_b_results": phase_b_results, "/feasibility": feasibility, "/results": results},
)
def calibration():
    os.chdir("/opt/uniqtoken")
    local = Path("/tmp/phase-c-cpu-calibration-v1/receipt.json")
    process = subprocess.run(arguments(local), capture_output=True, text=True)
    print(process.stdout[-4000:], flush=True)
    if process.returncode:
        print(process.stderr[-4000:], flush=True)
        raise RuntimeError(f"CPU calibration failed with exit {process.returncode}")
    receipt = json.loads(local.read_text())
    if receipt["experiments_started"] or receipt["validation_scored"] or receipt["test_split_opened"]:
        raise RuntimeError("CPU calibration scope violation")
    target = Path("/results/phase-c-cpu-calibration-v1/receipt.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError("immutable CPU calibration receipt already exists")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(local.read_bytes())
    os.rename(temporary, target)
    results.commit()
    return {"status": receipt["status"], "content_sha256": receipt["content_sha256"],
            "projection": receipt["projection"], "wall_seconds": receipt["wall_seconds"]}


@app.local_entrypoint()
def main():
    print(json.dumps(calibration.remote(), indent=2))
