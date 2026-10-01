# MergeEngine execution contract

Design proposal for [#107](https://github.com/umran666/UniqToken/issues/107)
(FBC-1), grounded in production source at commit
[`0718891`](https://github.com/umran666/UniqToken/commit/0718891f3f830b51318a1aedc865c4f99f3dfab5)
(2026-10-01). The API sketches below are internal design, not new
exports, implemented types, or a performance claim. Maintainer acceptance of
this document is the design review gate for the implementation track.

## Decision and scope

Separate merge execution from candidate learning and output reconstruction.
The first supported semantic profile is `SuperBpePassV1`: exact current
SuperBPE cross-word application, including left-to-right passes and dropout.
A generic boundary does not imply that all merge algorithms are interchangeable.
In particular, a rank-first BPE heap is not an equivalent implementation.

This change authorizes no runtime implementation, default-engine selection,
public Python/Rust/C/WASM API, model format, vocabulary, scoring objective, or
serialization change. Reference implementation belongs to
[#108](https://github.com/umran666/UniqToken/issues/108), the incremental Rust
prototype to [#109](https://github.com/umran666/UniqToken/issues/109), and
objective research to [#87](https://github.com/umran666/UniqToken/issues/87).

## Ownership

| Owner | Responsibilities | Outside its responsibility |
| --- | --- | --- |
| Text preparation and Unigram/Viterbi | Security policy, normalization/alignment, pre-tokenization, initial pieces, byte fallback, Unigram sampling | Runtime cross-word merge scheduling |
| CEM/SuperBPE learning and table adapter | Candidate generation, objective/score/rank definition, vocabulary growth, legality policy, immutable execution table | Executing merges or altering offsets |
| Caller/constraint adapter | Select semantic profile, supply stream boundaries and compiled legality decisions, validate configuration | Recomputing scores during execution |
| MergeEngine | Find live eligible adjacent pairs, schedule passes, apply supplied rules, invalidate candidates and dropout blocks, return ordered provenance | Learning, tokenization, decoding, creating IDs, normalizing text |
| Output adapter | Project pieces/IDs and source spans into existing API objects; retain unknown-ID, byte and special-token behavior | Choosing merges |
| Caller-owned randomness | RNG algorithm/state, probability validation, decision stream and row scheduling | Choosing candidate visitation order |

The engine evaluates supplied legality, but never invents legality policy.
Scoring remains upstream even if a future table includes ranks. Unigram
sampling and merge dropout are separate random consumers; the merge engine
starts only after initial segmentation has finished.

## Production behavior to preserve

The source of truth is
[`CustomTokenizer._apply_cross_word_merges` and its span counterpart](../uniqtoken/tokenizer.py),
not the CEM training heap or
[`BPEModel._encode_word_heap`](../uniqtoken/bpe_model.py).

1. `_cross_word_tokens()` selects vocabulary strings `t` satisfying
   `space_char in t[1:] and t.strip(space_char)`. This includes pieces with
   both a leading and an internal metaspace, but excludes leading-only and
   all-metaspace pieces. Apply the actual configured marker, not a hardcoded
   underscore or U+2581.
2. A pair is eligible when its **concatenated piece string** is in that set.
   The serialized Unigram model has vocabulary scores and IDs, not a replayable
   CEM pair history. `CrossEntropyMerging.merges` is learning-side metadata.
   Runtime accepts every reachable decomposition of an eligible result,
   not just the pair that originally learned it.
3. Scan the current pass left to right. Accept non-overlapping pairs, consuming
   two input elements at a time. A newly merged element cannot merge again
   until the next pass. Repeat only when the preceding pass accepted a merge.
4. No CEM score, vocabulary score, ID, rank, hash iteration order, or queue
   insertion counter determines runtime order. Competing overlapping pairs
   are resolved by the leftmost pair in the current pass.
5. For positive dropout, draw once for each eligible, unblocked pair actually
   visited. Drop when `random.random() < dropout_prob`. A dropped boundary
   stays blocked while both constituent pieces are unchanged. A neighboring
   accepted merge changes a constituent and releases that block. Unrelated
   merges and index shifts do not release it. Zero dropout consumes no draws.
6. The span path creates a merged `Token` with the concatenated piece,
   `model.token_to_id[piece]`, and `(left.raw_span[0], right.raw_span[1])`.
   Retained tokens keep their existing fields. Inputs are not mutated.

Training's exclusions for specials, byte pieces, maximum length and score
thresholds live in [`cem_merger.py`](../uniqtoken/cem_merger.py). The runtime
loop has **no additional special/byte/length filter**. Normal CEM tables exclude
those merges upstream, but a valid manually constructed vocabulary can admit
them. A reference adapter must preserve membership behavior even for such a
table; adding a protective runtime filter would change semantics.

Pre-tokenization chunks are flattened within each encode call, so word/chunk
boundaries are intentionally crossable. Separate calls and batch rows do not
share a stream. Production does not infer document boundaries from newline,
special-token spelling, or an arbitrary separator string. A new hard boundary
inside one call is an explicit experimental constraint, never a legacy default.

## Internal interface

The logical operation remains
`apply(tokens, merge_table, merge_constraints) -> tokens`. The internal form
adds an explicit per-call decision source and returns provenance so every
output mode can reconstruct its existing representation:

```text
MergeEngine.apply(
    input: Sequence<Atom>,
    table: MergeTableView,
    constraints: MergeConstraints,
    decisions: Optional<DropoutDecisions>,
) -> Result<MergePlan, MergeError>

Atom {
    piece: PieceKey,
    // Leaf number is its index in input. Source payload stays with the caller.
}

MergeTableView {
    vocabulary_identity: Identity,
    resolve(left: PieceKey, right: PieceKey) -> Optional<MergeRule>,
}
MergeRule {
    result: PieceKey,
    result_id: TokenId,
    learned_rank: Optional<NonNegativeInteger>,
    // Scores may be retained by the adapter for diagnostics, never evaluated.
}

MergeConstraints {
    semantic_profile: SuperBpePassV1,
    vocabulary_identity: Identity,
    hard_cuts: SortedSet<LeafBoundary>,
    legality: ImmutableLegalityView,
}
ImmutableLegalityView.allows(left, right, rule, leaf_range) -> bool
DropoutDecisions.drop_next() -> Result<bool, DecisionError>

MergePlan {
    groups: Sequence<Group>,
    applied_merges: NonNegativeInteger,
    decisions_consumed: NonNegativeInteger,
}
Group {
    leaves: HalfOpenRange<LeafIndex>,
    piece: PieceKey,
    merged_id: Optional<TokenId>,
}
```

`PieceKey` denotes an exact piece string in a caller-owned immutable dictionary
for this call, including initial out-of-vocabulary pieces. It is not a public
vocabulary ID. Identical strings have the same key; hash collisions must still
compare exact strings. No Unicode normalization, decoding, or metaspace
substitution occurs in lookup. Dictionary representation is private: Python
strings in the reference path, and optionally integer handles in Rust.
The dictionary covers all initial pieces and possible table results before
execution; table keys and input keys use that same dictionary. A compiled
legality view may be a borrowed Rust enum/table rather than a callback across
FFI for each candidate, provided it gives the same supplied decisions.
Empty piece strings are representable; termination depends on leaf count,
not string length. Unsupported Python string representations (for example,
surrogates that cannot be borrowed as Rust UTF-8) must be classified explicitly
by a native prototype, not replaced with lossy text.

`TokenId` preserves the model's non-negative integer value, including sparse
IDs. A narrower native representation must reject values it cannot represent.
Each result belongs to the same immutable vocabulary and maps to exactly the
concatenation of its inputs and its existing ID. Singleton groups have
`merged_id = None`; projection retains the initial payload/ID, including any
already-resolved unknown-ID fallback. A group of two or more leaves carries
the final rule's result ID. The engine neither reassigns IDs nor requires an
initial unknown piece to be present in the vocabulary.

`Group.leaves` refers to original **token indices**, not text offsets. Groups
are nonempty, ordered, adjacent, and partition `[0, len(input))`. On merging
`[a, b)` and `[b, c)`, the new group is `[a, c)`. Its piece is the exact
concatenation of all leaf pieces. Thus `applied_merges == len(input) -
len(groups)`, independent of the schedule. Empty input returns an empty plan.

### Table and rank rules

The legacy table adapter resolves a pair by concatenated-string membership
in `_cross_word_tokens()` and returns the existing vocabulary ID. It need not
enumerate every decomposition or allocate a concatenated string on each
lookup; an implementation may compare the two borrowed strings against an
indexed result. Such a lookup must have exactly the same membership answers.

The legacy adapter supplies `learned_rank = None`. If an experimental adapter
supplies integer ranks, `SuperBpePassV1` accepts them as metadata and ignores
them for scheduling. Equal ranks on different pairs are allowed; pass position
still resolves overlap. A rank-priority semantic profile is **unsupported** by
this proposal and cannot be selected implicitly by supplying ranks. Float
CEM scores must not be converted into integer execution ranks by the engine.

An explicit pair-table builder may coalesce identical duplicate records.
Conflicting records for the same pair (different result, ID, or rank), invalid
ranks, unknown result pieces, mismatched IDs, and non-concatenating results
are configuration errors. Hash-map overwrite order is never conflict policy.
This is validation of a proposed internal table, not a new validation rule for
the existing public tokenizer. Absent rules are normal non-candidates.

### Constraints and boundaries

The production-compatible constraint adapter has no interior hard cuts and
an always-true additional legality view: existing membership encodes eligibility.
It must use the same model/marker identity as the table. Existing model cache
invalidation uses `(id(model), model._state_version, space_char)`; an internal
compiled-table cache must likewise invalidate on any of these changes.

An experimental hard cut `k`, with `0 < k < len(input)`, marks the gap between
leaves `k - 1` and `k`. A proposed merged interval `[a, c)` is forbidden if
`a < k < c`. Cuts do not shift when tokens merge. Each batch row is a separate
call, not a concatenated sequence with guessed cuts. Duplicate cuts coalesce;
out-of-range cuts are invalid configuration.

Additional legality is a pure supplied predicate over live operands, the
resolved rule, and the proposed leaf interval. It may encode already-decided
special/byte or other boundary policy. It cannot consult RNG, mutate the table,
recompute an objective, or change as unrelated pairs merge. Evaluate table
membership, hard cuts, and legality before consuming a dropout decision.
Rejected pairs remain unmerged and consume no decision. The adapter owns
constraint meaning; an engine unable to implement a requested predicate or
profile returns `UnsupportedSemantics`, never ignores it.

### Schedule and candidate invalidation

For `SuperBpePassV1`, the following pass loop is normative:

```text
current := singleton groups; blocked := empty
repeat:
    next := empty; accepted := 0; cursor := 0
    while cursor < length(current):
        left := current[cursor]; right := current[cursor + 1] if present
        if right exists, their live boundary is not blocked,
           and table/cuts/legality admit their concatenation:
            if decisions exists and decisions.drop_next():
                block this pair of unchanged constituents
                append left; advance cursor by 1
            else:
                append their merged group; advance cursor by 2
                invalidate their shared and both neighboring boundary blocks
                accepted += 1
        else:
            append left; advance cursor by 1
    if accepted == 0: return next
    current := next; preserve surviving blocks at their remapped boundaries
```

This is finite: each accepting pass decreases leaf-group count and a
non-accepting pass terminates. A dropped left pair does not prevent its right
constituent from merging with the next token later in the **same** pass.

A Rust implementation may use an arena of nodes, adjacency indices and a
queue. A queue candidate must identify both nodes, their generations, their
current adjacency, and its eligible pass. Before evaluating legality or RNG,
discard candidates with a dead node, changed generation, broken adjacency,
or an obsolete pass. Revalidate rule identity on live pieces. Stale entries
consume no decisions, increment no merge counts, and do not unblock anything.
Blocked dropout state is keyed by the two unchanged constituent identities
and generations; moving positions alone does not invalidate it.

All nodes created by an accepted merge are eligible as operands only in the
next pass. Untouched nodes can still merge later in the current pass. The
queue must choose the leftmost eligible pair of that pass, never learned rank
or insertion order. Local recomputation is allowed only if it produces this
exact visitation schedule, including candidates after dropout. A generation
counter must not wrap into a still-live candidate identity. Queue/storage
details and any complexity or speed claims remain prototype work.

### Randomness

`decisions = None` means zero dropout and no RNG access. For positive dropout,
the production adapter provides `drop_next()` using the existing Python
`random.random() < p`. It preserves global RNG state and the exact number and
order of draws. The caller continues to validate `p` using
`validate_dropout_prob`, including its existing error message and validation
before empty-input returns. Booleans, nonnumeric values, NaN, infinities, and
values outside `[0, 1)` remain invalid.

For differential testing, supply independent cursors over the same recorded
Boolean tape to each engine. Exhaustion is `DecisionError`, never a fresh RNG
draw. Also compare decision count/trace: final pieces alone can hide a retry
bug. A seed in a Rust RNG is not equivalent to a seed in Python's RNG. A native
prototype without compatible decision consumption must reject stochastic
execution. Shared global RNG with parallel batch workers remains scheduling
dependent as documented today; this design adds no per-row seed guarantee.

### Output, character offsets, and byte spans

The caller retains each leaf's original piece, ID, and optional source payload.
String output projects `Group.piece`; ID output retains singleton IDs and uses
`merged_id` for merged groups. The facade's existing `encode_to_ids` unknown
lookup (`unk_id`, falling back to `0`) stays outside the engine.

For existing `Token` output, retain singleton fields and reconstruct a merged
group `[a, b)` as `(input[a].raw_span[0], input[b - 1].raw_span[1])`, with the
group's piece and result ID. These are half-open **raw Python character**
offsets. Normalized character spans, raw-character alignment, byte offsets,
and internal leaf ranges must never be conflated. Do not derive offsets by
the merged piece's length or by decoding it.

`Token` currently has no byte-span field. The boundary supports an opaque
caller-owned byte-span payload without adding one publicly. If a future
consumer already has raw byte spans, projection preserves singleton spans
and uses their first start/last end for merged groups under that consumer's
existing contract. If it derives bytes from raw character spans, it must use
its original source and encoding policy; for valid Unicode this can use a
raw UTF-8 prefix-offset map. Neither normalized UTF-8 lengths nor piece
spellings (such as `<0xC3>`) are raw source lengths. Do not impose a new UTF-8
encoding requirement on surrogateescape inputs.

Byte fallback may emit several tokens for one character, each with the same
source span. Normalization can also expand characters or collapse whitespace.
There is no general requirement that source spans be disjoint or tile the
original text. The partition invariant applies to **leaves**, not raw spans.
Preserve exact endpoints and overlapping spans; do not snap, deduplicate,
or reinterpret them in the engine.

Byte token values/order and special token policy remain with preparation,
the supplied table, and output/decode adapters. Decode continues to use
`UnigramModel.decode`/`ByteFallbackEngine.decode_tokens`, followed by existing
indent decompression and escaped-metaspace restoration. This document makes
no stronger round-trip claim than the configured normalization already allows.

### Errors, borrowing, and mutation

| Condition | Required behavior |
| --- | --- |
| Empty/singleton input, empty table, absent rule, denied boundary | Successful unchanged groups; no draws |
| Invalid public text, probability, model, security configuration, or disconnected lattice | Existing validation order, exception type/message and failure location remain upstream |
| Invalid internal key, rule, cut, or table/constraint identity | `InvalidConfiguration` before execution or RNG consumption |
| Unsupported profile, legality view, ID width, string representation, or decision capability | `UnsupportedSemantics` before execution; explicit capability check |
| Exhausted/failing decision source | `DecisionError`; no partial plan returned |
| Internal adjacency/partition corruption | `InvariantViolation`; never a successful alternate tokenization |

The new internal error names are proposed classifications, not new public
exceptions. The reference wrapper preserves current facade errors unchanged.
Malformed direct calls to today's private helpers are not a new public
contract; do not change their behavior as part of introducing a wrapper.
Expected unsupported inputs are checked before dispatch. In experimental
FastMergeEngine mode, unsupported execution must fail visibly, not silently
run the reference engine. Explicit reference selection remains available.

An illustrative private Rust shape is:

```rust
trait MergeEngine {
    fn apply(
        &self,
        input: &[Atom],
        table: &MergeTable,
        constraints: &MergeConstraints,
        decisions: Option<&mut dyn DropoutDecisions>,
    ) -> Result<MergePlan, MergeError>;
}
```

The caller owns input, immutable dictionary/table/constraints and source
payloads throughout the call. The plan owns its output vector and handles
that remain valid against that dictionary; it retains no borrowed reference
after the caller releases the dictionary. Only call-local scratch adjacency,
queue, blocked state and the exclusive decision cursor mutate. No model,
input list, input `Token`, or shared constraint is mutated. Cache snapshots
must be immutable across worker calls; mutation/replacement between calls
invalidates them. Concurrent model mutation during a call is unsupported and
must not be made safe by reading a mixture of versions.

The engine must not retain Python objects, source text, borrowed pointers,
or decision cursors after returning. Memory failure returns no partial plan
where recoverable; consumed external RNG decisions cannot be rolled back.
This design makes no new recoverability promise for process-level allocator
failure. Rust arenas, lifetimes, node generations and PyO3 types stay private;
no trait, selector or plan type is exported by this issue.

## Reference and optimized paths

For #108, start with an adapter delegating production-compatible requests to
the current two private helpers. The legacy adapter uses the same tokenizer,
model snapshot, empty cuts, membership table, and Python RNG. It preserves
the no-table passthrough and current returned values. A plan-producing bridge
can call the span helper with synthetic `Token.raw_span = (i, i + 1)` leaf
ranges and initial payload IDs, then project real source payloads separately.
Those synthetic indices must never be exposed as text offsets. Compare this
bridge with **both** unmodified helpers, including empty pieces and fallback.

The initial delegation adapter cannot apply arbitrary new legality views,
cuts, or decision-source protocols: classify them as unsupported. #108 may
faithfully express the same pass loop over groups to support those requests,
with controlled differential tests against the original production path.
The oracle remains simple and readable; the production helpers must not be
removed before that parity evidence exists.

For #109, use the same logical table, profile, constraints and provenance
projection in Rust. Exact concatenation lookup and indexed node storage may
avoid repeated string construction and scans, but gains are a hypothesis.
`BPEModel` is local data-structure prior art, not the SuperBPE oracle; do not
copy, vendor or translate FastBPE source. A prototype reports its supported
capabilities and rejects the rest. There is no automatic production dispatch
or normal-operation fallback introduced here.

## Semantic review vectors

In this table `S` denotes the configured U+2581 marker, `A = "a"`,
`B = "Sb"`, `C = "Sc"`, `D = "Sd"`; concatenation denotes piece concatenation.
Eligible results are a vocabulary-membership table. Draws are numeric Python
RNG values with `p = 0.5`, so `0.0` drops and `0.99` accepts.

| Case | Input / eligible results / draws | Expected output and reason |
| --- | --- | --- |
| Empty and singleton | `[]` or `[A]`, any table | Unchanged, zero merges/draws |
| Leftmost overlap | `[A,B,C]`, `{AB,BC}` | `[AB,C]`, irrespective of scores/ranks |
| Pass barrier | `[A,B,C,D]`, `{AB,BC,ABC,CD}` | `[AB,CD]`; eager `AB -> ABC` would incorrectly steal `C` from `CD` |
| Hierarchy | `[A,B,C]`, `{AB,ABC}` | `[ABC]` after two passes |
| Different decomposition | `[AB,C]` or `[A,BC]`, `{ABC}` | Both produce `[ABC]`; learning history is irrelevant |
| Persistent dropout | `[A,B,C,D]`, `{AB,CD}`, draws `[0.0,0.99]` | `[A,B,CD]`, exactly two draws; unrelated merge cannot retry `AB` |
| Changed constituent | `[A,B,C]`, `{AB,BC,ABC}`, draws `[0.0,0.99,0.99]` | `[ABC]`, three draws; `BC` releases the old `AB` block |
| Marker membership | `['S','b']`, vocabulary `{'Sb'}` | Unchanged: leading-only marker is not eligible |
| Leading plus internal marker | `['Sa','Sb']`, vocabulary `{'SaSb'}` | `['SaSb']` |
| Custom special/byte table | `['<\|s\|>',B]` or `['<0x61>',B]`, their concatenations in vocab | Membership may merge them; do not add a filter |
| Empty piece | `[AB,'']`, `{AB}` | `[AB]` with leaf range `[0,2)`; even unchanged piece text can consume a leaf |
| Shared source spans | Two byte fallback leaves for one character | Preserve both leaves/spans absent a supplied rule; no overlap rejection |
| Hard cut / denied legality | `[A,B]`, `{AB}`, cut `1` or predicate false | Unchanged with zero draws; experimental constraint, not production default |
| Stale candidate | Candidate refers to `B` after `AB` consumed it | Discard without a draw, merge, or block change |

## Validation and review gates

Existing baseline tests supporting this design:

```bash
python -m unittest tests.test_tokenizer.SuperBPETests tests.test_tokenizer.CrossEntropyMergingTests tests.test_subword_regularization tests.test_tokenizer.NormalizerTests tests.test_tokenizer.ByteFallbackTests tests.test_tokenizer.CustomTokenizerTests tests.test_tokenizer.SecurityAndIndentationTests -q
```

At the pinned source this passed 55 tests on Windows/Python 3.10. This verifies
existing behavior only; it is not proof of a future engine's parity.

This runnable Python probe checks the central scheduling examples against
both unchanged production helpers, including leaf-range projection and exact
RNG draw counts. It implements no MergeEngine and uses no training corpus:

```python
from unittest.mock import patch

from uniqtoken.pre_tokenizer import Normalizer, RegexPreTokenizer
from uniqtoken.tokenizer import CustomTokenizer, Token
from uniqtoken.unigram_trainer import UnigramModel

A, B, C, D = "a", "\u2581b", "\u2581c", "\u2581d"
cases = [
    ([A, B, C], [A + B, B + C], [A + B, C], 0.0, []),
    ([A, B, C, D], [A + B, B + C, A + B + C, C + D], [A + B, C + D], 0.0, []),
    ([A, B, C], [A + B, A + B + C], [A + B + C], 0.0, []),
    ([A, B, C, D], [A + B, C + D], [A, B, C + D], 0.5, [0.0, 0.99]),
    ([A, B, C], [A + B, B + C, A + B + C], [A + B + C], 0.5, [0.0, 0.99, 0.99]),
    ([A + B, ""], [A + B], [A + B], 0.0, []),
]
for pieces, results, expected, probability, draws in cases:
    vocab = {piece: -1.0 for piece in dict.fromkeys(pieces + results)}
    ids = {piece: i for i, piece in enumerate(vocab)}
    model = UnigramModel(
        vocab=vocab,
        token_to_id=ids,
        id_to_token={i: piece for piece, i in ids.items()},
        special_tokens=[],
        max_subword_len=32,
        byte_fallback=False,
    )
    tok = CustomTokenizer(Normalizer(normalize_unicode=False), RegexPreTokenizer(), model)
    with patch("random.random", side_effect=draws) as rng:
        assert tok._apply_cross_word_merges(pieces, probability) == expected
    assert rng.call_count == len(draws)
    leaves = [Token(piece, ids[piece], (i, i + 1)) for i, piece in enumerate(pieces)]
    with patch("random.random", side_effect=draws) as rng:
        groups = tok._apply_cross_word_merges_with_spans(leaves, probability)
    assert rng.call_count == len(draws)
    assert [group.text for group in groups] == expected
    cursor = 0
    for group in groups:
        start, end = group.raw_span
        assert start == cursor and end > start
        assert group.text == "".join(pieces[start:end])
        assert group.id == ids[group.text]
        cursor = end
    assert cursor == len(pieces)
print("6 contract probes passed")
```

All six probes passed at the pinned source. The ordering/dropout review
vectors above must additionally become focused oracle
tests in #108. The full differential suite belongs to
[#110](https://github.com/umran666/UniqToken/issues/110), including fixed seeds,
random/adversarial tables and tapes, conflicting rules, offsets/byte payloads,
counts, errors, reduced failures, and platform/build coverage. Full semantic
coverage is [#114](https://github.com/umran666/UniqToken/issues/114).

| #107 acceptance criterion | Design evidence / review question |
| --- | --- |
| Reviewed document with invariants | This proposal; maintainer PR review is still required. Do pass barriers and block lifetimes match the source? |
| Clear scoring, legality, execution, reconstruction, randomness ownership | Ownership table and interface. Is any objective computation leaking into execution? |
| Existing implementation can be wrapped unchanged | Legacy delegation/leaf-span bridge above; differential implementation is #108. Are custom-table and RNG cases preserved? |
| Efficient Rust boundary without public Rust details | Borrowed immutable inputs, integer piece handles, owned plans, private arena/generations. Does the prototype preserve pass order? |
| IDs, pieces, offsets, bytes, fallback, specials and decode unchanged | Projection rules and legacy membership, shared spans and errors. Are source coordinates preserved exactly? |
| No default/public API change | Documentation only; no engine code, exports, selector, or model-format changes |

After #108/#109, require zero mismatches in #110 before controlled benchmarking
in [#111](https://github.com/umran666/UniqToken/issues/111). Queue/allocation
optimization (#112/#113) follows measurements. Experimental integration
[#115](https://github.com/umran666/UniqToken/issues/115) requires differential
and full semantic evidence and keeps the current default. Production adoption
is a separate decision in
[#116](https://github.com/umran666/UniqToken/issues/116). This proposal neither
asserts a speedup nor preselects adoption.
