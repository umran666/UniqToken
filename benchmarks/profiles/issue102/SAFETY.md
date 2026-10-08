# Issue #102: NFKC fast-path invariant

This invariant and validation domain are specified before the runtime change.
The baseline is `a0fd66636051dbf03b83525a8557da335949c39b` (the current main).
Only synthetic engineering fixtures are used; no research data or Phase C work
is part of this change.

## Predicate and proof

Let `N(s)` be the existing Rust `s.nfkc().collect::<String>()` and let `A(s)`
mean every UTF-8 byte of `s` is ASCII. The proposed helper returns a borrowed
`s` when `A(s)` and an owned `N(s)` otherwise. Every ASCII character has an
identity compatibility decomposition and canonical combining class zero;
ASCII sequences cannot form a canonical composition pair. Therefore
`A(s) => N(s) = s`, including empty strings, controls, CR/LF and DEL. For
non-ASCII input the exact existing normalization iterator remains in use.
The helper's text is consequently equal to `N(s)` for every valid Rust string.

This is intentionally a fast path for the ASCII subset of already-NFKC text.
Already-normalized non-ASCII text still follows the existing full path. It
does not use a speculative per-character identity test or treat Unicode
QuickCheck `Maybe` as normalized.

## Observational equivalence

- **Text:** replace only the NFKC intermediate. Unicode-space conversion,
  punctuation mapping, lowercasing, whitespace collapse/strip and metaspace
  escaping still execute with the same flags, ordering and validation.
- **Alignment:** the existing `normalized == text` branch already assigns
  `(i, i + 1)` to each character. The borrowed ASCII result takes that same
  branch. Non-ASCII normalization and the prefix-based span algorithm remain
  unchanged, even when the non-ASCII input is already NFKC.
- **Pretokenization, graphemes, IDs and offsets:** these consume the same
  normalized text and alignment; their algorithms and model are unchanged.
- **Security/special tokens:** the native security gate still rejects private
  escape characters and checks `"<|"` on the canonical NFKC text regardless of
  the normalizer's `normalize_unicode` flag. ASCII control syntax is checked
  directly because it is already canonical; non-ASCII synthesized delimiters
  still undergo full NFKC. Python security policy remains unchanged.
- **Errors:** Python argument conversion, surrogate handling, space-character
  validation and C ABI UTF-8 validation remain ahead of/around the same calls.
  No new unchecked conversion, public entry point or accepted input type is
  introduced. Allocation-failure timing is not an API guarantee.
- **Ownership:** borrowed input is used synchronously within the call. Public
  return values remain owned strings/lists, with no lifetime crossing the FFI.

## Defined differential domain

1. Rust helper versus the original NFKC iterator: every Unicode scalar value
   (all code points except surrogates), both alone and in combining, delimiter,
   CR/LF and emoji/ZWJ contexts. This includes unassigned values, noncharacters,
   all general categories, modern scripts and Hangul. Also every ASCII string
   of length zero, one or two.
2. Before/after native binaries: normalized text and source alignment under
   all 64 combinations of the six native normalizer flags, with synthetic
   ASCII controls/whitespace, multilingual text, combining/reordering,
   compatibility expansions, Hangul, emoji/ZWJ, literal metaspace/escape
   characters and security delimiter cases. Include custom and invalid
   space characters and unsupported Python inputs.
3. Before/after public API: tokens, IDs, source offsets, pretokenization,
   single/batch paths, decoding and security policies on those same text
   cases. Existing Unicode, native, byte-fallback, security and C ABI tests
   remain part of validation, including malformed UTF-8 at the C boundary.

The proof is reviewed against each call site and backed by this bounded,
reproducible domain; it is not a claim of exhaustive testing of all strings.

## Measurement gate

The retained #95 report measures normalization at 3.312 ms and the NFKC
security gate at 2.720 ms on `long_batch` (190,240 normalized UTF-8 bytes),
versus 74.449 ms end-to-end. These diagnostic row sums are not additive wall
time and regex remains dominant. Targeted before/after measurements must
repeat the fixed fixtures on current main, include `is_ascii()` detection
inside the timed production path, and retain raw paired trials. Report
normalized-byte MB/s, build/thread/environment fingerprints, output hashes
and within-run uncertainty. Include early/late non-normalized inputs to
check detection overhead. No universal speedup or cross-tokenizer superiority
is implied.
