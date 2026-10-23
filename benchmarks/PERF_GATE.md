# Correctness and performance gates (#106)

The gate consumes the existing #105 canonical benchmark receipts. Correctness
is authoritative. Performance is advisory until a separately reviewed
measurement protocol and controlled repeat-run evidence justify enforcement.
No historical benchmark receipt or frozen research artifact is rewritten.

## CI behavior

`.github/workflows/perf-regression-gate.yml` runs on relevant pull requests,
weekly on main, and by manual dispatch.

- `tokenizer-parity` runs deterministic exact parity. A mismatch fails the job.
- `performance-gate` runs the canonical repeated benchmark, validates its receipt,
  compares a versioned baseline and uploads the comparison and source artifacts.
- Invalid current input exits **2**, with a diagnostic artifact when `--out` is
  supplied. Missing, stale or unsuitable baseline evidence is an explicit soft
  skip. Neither outcome is reported as a successful performance comparison.
- `--enforce` records an enforcement request for CLI compatibility. Canonical
  schema **1.0 is not eligible**: `enforcement_eligible` is false and throughput
  or memory movement cannot become a merge-blocking performance verdict.

## Interpretation and limits

Throughput is normalized UTF-8 input decimal MB/s; peak memory is `peak_rss_mb`.
The candidate thresholds remain **5% throughput drop** and **10% peak RSS rise**.
They are investigation signals, not calibrated acceptance thresholds.

Despite its name, #105 field `throughput_ci_95_mb_s` contains the minimum and
maximum throughput observed across repetitions. These are **observed extrema**,
not an estimated 95% confidence interval. The gate reports `outside-observed-range`
or `nominal-regression`; neither proves statistical significance. Peak RSS has
no repeated-memory uncertainty in this schema. Even matching hardware and
non-overlapping extrema cannot establish the missing calibration.

#106 remains incomplete on statistical enforcement: an approved uncertainty
estimator, independent repeat-run noise characterization and memory evidence
are required before a later change enables it. The existing receipt schema and
benchmark methodology are preserved here.

## Input and baseline integrity

Current input must use schema 1.0 and contain unique workload objects, positive
finite throughput/memory, ordered finite throughput bounds containing the
reported throughput, and valid fixture/batch identities. Incomplete or changed
workload matrices produce `workload-mismatch`, never `pass`.

A baseline must link an available SHA-256-verified source receipt. Its workload
records must agree with that source. Missing provenance, malformed baseline
metadata, invalid lifetime or expired evidence prevents comparison. Measurement
and recording timestamps are timezone-aware; the default lifetime is 30 days.

Comparability includes platform, processor, physical/logical CPUs, RAM (rounded
to GB), Python, build mode, worker count, warmups, repetitions, iterations and
package/backend identities. Candidate Git commits may differ. Artifact hashes
stay recorded in the source receipt for provenance but are not comparability
keys: SentencePiece shuffles its training corpus, so a retrained canonical
artifact does not reproduce the same model bytes. All workloads must retain
consistent repeated-measurement counts; at least three observations are
required as an evidence sanity check, which alone supplies no statistical
calibration. Missing fingerprint fields are not considered a controlled match,
even when absent on both sides.

The committed baseline is derived deterministically with `--write-baseline`
from the reviewed #105 receipt; no measurement value is hand-edited or invented.
Its 30-day lifetime is anchored at the source measurement time. Regenerate it
only from a reviewed receipt.

## Commands and triage

```sh
python -m benchmarks.canonical_gate --results perf-gate/current/results.json --baseline benchmarks/canonical/gate-baseline.json --out perf-gate/comparison.json
```

To record a new baseline from a trusted, reviewed source receipt:

```sh
python -m benchmarks.canonical_gate --write-baseline --results /path/to/new-results.json --baseline /path/to/new-baseline.json
```

Inspect `status`, `note`, provenance and fingerprint differences before the
per-workload signals. Invalid input must be repaired; unsuitable baseline evidence
must be refreshed with a documented run. Retain the original evidence.

Repeat suspect measurements on the same controlled host with the same protocol,
retain every workload, and investigate code changes. Repeated observed-range
separation is a reason to investigate, not a calibrated significance test. A
confirmed regression can be fixed or reverted without weakening exact parity.

Local decision-logic checks:

```sh
python -m unittest tests.test_canonical_gate -v
```
