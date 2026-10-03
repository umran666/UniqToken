# Native performance evidence for issues 96-99

The combined prefix-visitor and compact decoder is the proposed implementation.
The prefix-only version is an attribution ablation; two of its batch cells have
repeatable losses and it is not proposed as a standalone release. The combined
version has no repeatable degraded batch cell, including batch size one.

Retained evidence is under `benchmarks/native_performance/issue96-99`:

- `summary/REPORT.md` states methods, uncertainty, memory scope and limits.
- `summary/metrics.csv` and `summary/summary.json` contain all 291 variant/cell rows.
- `summary/batch-span-parity.json` verifies all 48 raw single/batch span cells.
- Six PNG plots cover segmentation, full batch throughput/allocation/memory and
  dense-prefix scaling; each is generated from the retained measurements.
- Three receipt ZIPs preserve every original request, response, result and
  manifest byte for the baseline profile, paired timing and prefix allocation
  attribution. Archive manifests and every contained file hash are checked by
  `native_receipt_archive.unpack` before extraction. ZIP timestamps are fixed.
- The outer manifest binds archives and summary artifacts, retaining separate
  measurement, report-export and archive-publication identities.

Across the 45 natural-script/vocabulary segmentation cells, combined median
paired speedups range from 2.45x to 9.13x. Every segmentation cell, including
the four dense-prefix stress cells, has interval bounds above one in both
independent worker rounds. Batch results include 22 repeatable gains and 26
inconclusive/mixed cells, with no repeatable degradation. These are local fixed
fixtures, not a production-corpus or hardware-population claim.

For 256 Latin characters with the 8K model, decoder allocation requests change
from 2370 to 2114 for the prefix-only ablation, then to 109 for the combined
version. Requested allocation bytes change from 209879 to 111575 to 15624.
Temporary Rust live-byte peaks are 103568, 103568 and 15624 respectively: the
prefix change reduces allocation volume while the compact lattice also lowers
peak scratch. The trie representation and vocabulary are identical.

Raw segmentation allocation counts cover the uncached Rust decoder before
Python materialization, while primary timings include the ordinary raw API
wrapper. Full-batch counts wrap the ordinary fused Rust API. CPU and process
RSS scopes and Windows accounting limits are explicit in the report. No extra
cache, pooling, trie representation, normalization or security change is made.

The dense fixture uses all `a` prefixes of lengths 1-16 with score `-length`,
exercising exact score ties. Control/byte weights are -10. These synthetic
finite weights are deliberately not a trained, probability-normalized model.
Its length sweep demonstrates observed scaling for fixed maximum length 16;
the source establishes O(n L) time and O(n) scratch/output bounds. Unbounded
maximum length permits quadratic time. Pruned decoding retains its old lattice.

Build the three source versions named in the report with identical release
flags (opt-level 3, LTO, one codegen unit). Preserve default and allocation-feature
extensions in distinct directories. Primary measurements require an otherwise
idle host; allocation instrumentation is absent from default wheels.

```text
python -m benchmarks.compare_residual_native --baseline BASE_DLL --prefix PREFIX_DLL --compact COMPACT_DLL --baseline-allocation BASE_ALLOC_DLL --compact-allocation COMPACT_ALLOC_DLL --models benchmarks/byte_fallback/issue86 --baseline-commit BASE_SHA --prefix-commit PREFIX_SHA --compact-commit COMPACT_SHA --output NEW_PAIRED_DIR
python -m benchmarks.profile_prefix_allocations --source NEW_PAIRED_DIR/results.json --native PREFIX_ALLOC_DLL --build-commit PREFIX_SHA --output NEW_PREFIX_DIR
python -m benchmarks.native_performance_report --source NEW_PAIRED_DIR/results.json --prefix-source NEW_PREFIX_DIR/results.json --verify-batch-spans --output NEW_SUMMARY_DIR
python -m benchmarks.native_receipt_archive --baseline BASE_PROFILE_DIR --paired NEW_PAIRED_DIR --prefix NEW_PREFIX_DIR --summary NEW_SUMMARY_DIR --output NEW_PUBLISHED_DIR
```

Each output must be new; commit source/protocol changes before measuring or
exporting. The retained-evidence test verifies all archive bytes, complete
matrices, streams/scores, uncertainty, memory attribution, batch-one gate and
summary recomputation. No corpus or held-out test data is read by these tools.
