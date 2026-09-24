# Phase C Status

**NOT EXECUTED — COMPUTE CONSTRAINED**

The Phase C confirmatory protocol was frozen before execution. It specifies an
unchanged 300,000,000-normalized-byte training exposure, three tokenizers
(SPM-Unigram, Boundary-BPE, and UT-SuperBPE), three new seeds, nine total LM
runs, and the frozen confirmatory analysis plan.

Phase C confirmation was not executed. Capped, non-experimental Modal
calibrations measured the actual training stack on the available configurations
and projected that the complete frozen grid, including the required 50 percent
operational allowance, exceeded the verified free workspace capacity. These
calibrations produced infrastructure evidence only: no Phase C condition
records, validation metrics, or confirmatory results.

FLORES-200 devtest remained unopened. Phase C was therefore not a scientific
failure, and no confirmatory claim is made. In particular, the exploratory 16K
byte-matched UT-SuperBPE Phase B result is not confirmed.

Legitimate additional compute could permit future execution of the already
frozen protocol. Such execution must preserve its exposure, tokenizer cohort,
seeds, architecture, stopping rule, test isolation, and statistical analysis;
this status is not authorization to change those requirements or launch a run.

The supporting measured calibration evidence is recorded in:

- `benchmarks/PHASE_C_ACCELERATOR_CALIBRATION_REPORT.md`
- `artifacts/modal-runtime-gate/receipts/phase-c-gpu-calibration-measurement.json`
- `artifacts/modal-runtime-gate/receipts/phase-c-cpu-calibration-measurement.json`
- `artifacts/modal-runtime-gate/receipts/phase-c-t4-calibration-measurement.json`
- `artifacts/modal-runtime-gate/receipts/phase-c-a100-40gb-calibration-measurement.json`
- `artifacts/modal-runtime-gate/receipts/phase-c-accelerator-calibration-independent-verification.json`
