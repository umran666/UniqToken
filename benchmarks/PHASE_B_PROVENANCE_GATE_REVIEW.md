# Phase B Provenance-Gate Review

Status: launch-blocking review. No frozen selection, exposure, tokenizer artifact, or benchmark result is modified here.

## `truncated_to_quota` trace

The flag is produced by `benchmarks/freeze_phase_a_dataset.py:write_partition`. When the next upstream record exceeds the remaining normalized-byte quota, `normalized_prefix()` selects a UTF-8-safe prefix and `source_record(..., truncated=True)` writes it into the frozen Phase A training corpus.

The Phase B exposure contains one such record:

- exposure position: `17`
- id: `allenai/MADLAD-400:data-v1p5/ru/clean_docs_v2-03174-of-05000.jsonl.gz:1611:13333250`
- frozen normalized/source bytes: `84` / `84`
- local upstream record at the pinned source file and record index: `2,049` UTF-8 bytes
- source manifest flag: `truncated_to_quota: true`

Therefore the record is a whole document only relative to the already quota-truncated Phase A corpus. It is not the complete upstream MADLAD document. The exact exposure remains internally consistent and its 1,000,000 normalized-byte hash is unchanged, but the original protocol language “whole-document inclusion” is not satisfied relative to the upstream source for this record.

### Gate decision

**Unresolved; launch blocked.** Do not remove the flag, edit the record, or regenerate the exposure while claiming the frozen exposure is unchanged. The choices requiring an explicit protocol decision are:

1. regenerate the source manifest and exposure from untruncated upstream documents, which changes the exposure provenance and requires a new frozen exposure hash; or
2. amend the protocol to define whole-document inclusion relative to the Phase A quota-truncated corpus and explicitly disclose that upstream records may be prefix-truncated.

No Phase B result should be collected until one choice is approved and versioned.

## Runtime and commit gates

The v6 preflight recorded a Windows runtime (`Python 3.10.11`, extension hash `870fbcf...`) that does not match the pinned Linux Phase A runtime (`Python 3.10.17`, extension hash `b98f262d...`). The pinned Linux/GPU environment has not yet been reproduced and fingerprinted in this workspace.

The exact read-only preflight implementation used for v6 is:

- `tools/phase_b_flop_preflight.py`
- `tools/verify_phase_b_flop_preflight.py`
- `tests/test_phase_b_flop_preflight.py`
- frozen selection/exposure/source/Phase A artifact paths and hashes recorded in `artifacts/phase-b-flop-preflight-v6/preflight-rejection.json`

The protocol amendment is committed as `f2e8d24bac96ac9483459e1a255b256f4b684c06`. The preflight implementation itself still needs a dedicated clean commit before launch; unrelated untracked files must remain untouched and need not be deleted.

## Operations gate

No durable Modal storage, resume ledger, hard time limit, or spending cap has been verified in this checkout. Modal configuration must be inspected and recorded only after the runtime and source-provenance decisions are complete. No GPU or LM experiment is authorized by this review.
