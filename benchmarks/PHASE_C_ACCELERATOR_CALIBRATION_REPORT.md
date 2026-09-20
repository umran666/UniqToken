# Phase C Accelerator Calibration Report

## Scope

This is non-experimental infrastructure calibration evidence. No Phase C
condition was started, no validation metric was computed, the held-out test set
was not opened, and no launch is authorized. The frozen Phase C protocol,
300 MB exposure, tokenizer cohort, seeds, architecture, budgets, stopping rules,
and analysis plan are unchanged.

Both probes used commit `cea54b22864823cf9aa054e410dee652451e5886`,
the same ordered 203-document / 999,753-normalized-byte training-only prefix,
and exactly 256 optimizer steps for each of SPM-Unigram, Boundary-BPE, and
UT-SuperBPE. The prefix hash was
`76c8f5a5701860c4291468f8bb145f168704bd93b474aaa08e4db4b7ec29f028`.

## Measurements

| GPU | Probe wall time | SPM-Unigram | Boundary-BPE | UT-SuperBPE | Peak allocated GPU memory | Modal app charge |
|---|---:|---:|---:|---:|---:|---:|
| NVIDIA T4 | 251.570379 s | 7.596070 s | 143.776163 s | 6.415034 s | 167,647,744 bytes | $0.07 |
| NVIDIA A100-SXM4-40GB | 150.672829 s | 5.577073 s | 91.164166 s | 4.762144 s | 167,647,744 bytes | $0.10 |

Each per-tokenizer time includes model/optimizer initialization, tokenization,
and 256 optimizer steps. Total probe wall time additionally includes frozen
provenance validation, tokenizer artifact loading, receipt construction, and
other orchestration inside the calibration process.

The pinned runtime was Python 3.10.17, PyTorch 2.6.0+cu124, CUDA 12.4,
SentencePiece 0.2.1, NumPy 2.2.6, regex 2026.4.4, NVIDIA driver 580.95.05,
and Rust extension signature
`b98f262df1c63e1b4fd0cfa38f5b673ce4affd8f8349bd6f642c13ef2c47db42`.
Both containers recorded `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
`RAYON_NUM_THREADS=1`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Projection And Decision

Modal's published pricing observed on 2026-09-20 was used: T4
$0.000164/s and A100 40 GB $0.000583/s, plus $0.0000131/core/s for four
physical CPU cores and $0.00000222/GiB/s for 16 GiB memory. The resulting
all-in rates were $0.906912/hour and $2.415312/hour, respectively.

| GPU | Point time | Point cost | Time with 50% allowance | Cost with 50% allowance | Capacity | Decision |
|---|---:|---:|---:|---:|---:|---|
| T4 | 63.786220 h | $57.848489 | 95.679330 h | $86.772733 | $13.60 | **FAIL** |
| A100-40GB | 42.726968 h | $103.198959 | 64.090452 h | $154.798438 | $13.60 | **FAIL** |

These are point extrapolations from the capped measurements, not lower bounds,
upper bounds, or guaranteed costs. The required 50% allowance covers unmeasured
validation, orchestration, checkpointing, and billing variance only as a
conservative planning factor; it is not a guarantee.

Workspace usage after both probes was $16.40 of the $30 hard usage limit,
leaving $13.60. Current post-credit charges were $0 against the $13 spend
limit. App charges and rounded workspace deltas are recorded independently in
the billing receipts.

The separate verifier recomputed the projections from the frozen feasibility
step counts and classified both GPUs as `FAIL`. Phase C remains blocked, and no
launch authorization was created.
