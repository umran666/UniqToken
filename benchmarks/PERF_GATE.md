# Performance Regression Gate Guide (Issue #106)

CI regression gate built on the canonical end-to-end benchmark ([#105](https://github.com/umran666/UniqToken/issues/105), `benchmarks/CANONICAL_BENCHMARK.md`). It separates hard correctness failures from statistical performance signals so that a noisy public shared runner can never block merges based on one timing sample.

## Gate Structure (correctness and performance are visibly separated)

Workflow: `.github/workflows/perf-regression-gate.yml` (pull requests to `main`/`master`, weekly schedule on `main`, and manual dispatch).

| Job | Signal | Behavior |
| :--- | :--- | :--- |
| `tokenizer-parity` | Deterministic exact-parity gate (`tests.test_canonical_benchmark.CanonicalBenchmarkTests.test_parity_gate_passes_on_canonical_model`, i.e. the same `run_parity_gate` the canonical benchmark runs) | **Authoritative. Any parity failure is a hard failure** and blocks merge. |
| `performance-gate` | Canonical benchmark measurements compared against the versioned baseline | Advisory by default. Merge-blocking only when explicitly enforced on controlled, fingerprint-matched hardware. |

The canonical benchmark additionally re-runs the exact parity gate in-process before any timing is recorded, so a parity failure aborts the performance run before numbers exist.

## Thresholds and Their Rationale (tied to measured variance)

Initial candidates from the issue: throughput drop > 5%, memory rise > 10%. These are **not** applied to a single sample:

- Throughput regressions are reported as *significant* (merge-blocking candidates) only when the nominal drop exceeds 5% **and** the current run's repeated-measurement 95% uncertainty interval (`throughput_ci_95_mb_s`, 7 repetitions per cell, from the #105 schema) does not overlap the baseline's interval. A current run whose best repetition still lands below the baseline's worst repetition is a real regression; anything inside the noise band is recorded as `nominal-regression` and stays advisory.
- Memory regressions (> 10% peak RSS rise) follow the controlled-environment rule: they fail only under `--enforce` on a fingerprint-matched runner, otherwise they are advisory.

Rationale from the published #105 receipts (`benchmarks/canonical/issue105/results.json`, 60 workload cells): the relative half-width of the per-cell throughput 95% range has **median 14.3%, mean 16.0%, p90 30.2%, max 40.4%**. A single nominal 5–8% throughput drop is therefore *well inside* the observed run-to-run noise of a shared runner; requiring non-overlapping repeated-measurement intervals is what gives the threshold statistical meaning.

Units and schema: throughput uses the canonical primary metric (normalized input decimal MB/s, 10^6 bytes/s) and memory uses `peak_rss_mb`, both exactly as emitted by the #105 `results.json` schema. The gate adds no new metric definitions.

## Baseline Provenance and Expiry Policy

Baseline artifact: `benchmarks/canonical/gate-baseline.json` (committed, versioned, `schema_version: "1.0"`).

- **Provenance**: records the source receipt path, its SHA-256, the git commit, and the measurement timestamp. The gate re-verifies the recorded digest against the in-repo receipt on every run; a mismatch yields advisory status `baseline-provenance-mismatch` (never silently substituted, never merge-blocking).
- **Expiry**: anchored at the source measurement time, 30-day default (`--max-age-days`). An expired baseline yields advisory status `baseline-expired`.
- **Fingerprint**: platform, machine, processor, physical/logical CPUs, and total RAM. Fingerprint mismatch yields advisory status `hardware-mismatch` — this is the defined behavior on unsuitable CI hardware: the run still measures, uploads artifacts, and records the differences, but cannot block a merge.

Recording a baseline from a trusted host:

```powershell
python -m benchmarks.canonical_benchmark --output benchmarks/canonical/issue106
python -m benchmarks.canonical_gate --write-baseline --results benchmarks/canonical/issue106/results.json --baseline benchmarks/canonical/gate-baseline.json
```

## Running the Gate Locally

```powershell
# Advisory comparison (default; safe on any machine)
python -m benchmarks.canonical_gate --results perf-gate/current/results.json --baseline benchmarks/canonical/gate-baseline.json --out perf-gate/comparison.json

# Enforced comparison (controlled hardware only)
python -m benchmarks.canonical_gate --results perf-gate/current/results.json --out perf-gate/comparison.json --enforce
```

Every run writes a machine-readable comparison artifact (`--out`) containing the status, thresholds, baseline provenance/expiry, fingerprint diff, and per-cell throughput/memory verdicts. Exit codes: `0` pass/advisory/soft-skip, `1` enforced significant regression, `2` hard input error.

## False-Positive Triage, Rerun, Baseline Update, Rollback

1. **Triage**: open the comparison artifact (`perf-gate/comparison.json`) and the uploaded `results.json`. Check `status` first:
   - `hardware-mismatch` / `baseline-expired` / `baseline-provenance-mismatch` / `fixture-drift` / `baseline-missing`: the gate could not compare validly. No action against the pull request; fix the baseline (see below). Fixture drift means the versioned #105 fixtures changed, which is a benchmark-integrity bug.
   - `advisory` with `nominal-regression` cells: noise-band movement. Comment and move on.
   - `advisory`/`fail` with `significant-regression` cells: candidate real throughput regression; investigate the diff before merging.
2. **Rerun**: re-run the `performance-gate` workflow (or `workflow_dispatch`). Because the significance rule requires non-overlapping repeated-measurement intervals, a one-off noisy run is expected to clear on rerun. Two independent significant runs should be treated as real.
3. **Baseline update**: when a change is an intentional improvement (or the 30-day expiry passes), re-record the baseline from a trusted host using the commands above and commit it in the same pull request. Never hand-edit the baseline records.
4. **Rollback**: if a significant regression is confirmed, revert or fix the offending change; the parity job remains authoritative throughout, so correctness is never traded against performance.

## Acceptance Criteria Mapping

- [x] CI integration with correctness and performance gates visibly separated (two named jobs).
- [x] Threshold rationale tied to measured variance (non-overlap rule, quantified above).
- [x] Deterministic exact-parity gate is authoritative (separate blocking job; aborts measurements).
- [x] Repeated performance measurements and uncertainty recorded (7 repetitions/cell in `results.json`; intervals in the comparison artifact).
- [x] False-positive triage, rerun, baseline-update, and rollback procedures documented (above).
- [x] Memory and throughput regressions use stable units and the canonical #105 schema.
