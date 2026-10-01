"""Reference MergeEngine correctness oracle for SuperBPE runtime merges.

Implements the internal MergeEngine contract from issue #107 / docs design
(``SuperBpePassV1``). This module is intentionally readable and is a
correctness oracle, not a performance target.

Ownership (preserved from the design):
- CEM/SuperBPE learning owns candidate scoring and vocabulary growth (#87).
- MergeEngine owns only deterministic application of a supplied merge table
  and constraints.
- Tokenizer output reconstruction (IDs, offsets, byte spans, fallback,
  specials, decode) stays with the caller / facade.

This module is an internal research surface. It does not change the default
tokenizer path or public package exports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

from .tokenizer import CustomTokenizer, Token

# ---------------------------------------------------------------------------
# Errors (internal classifications; not new public facade exceptions)
# ---------------------------------------------------------------------------


class MergeError(Exception):
    """Base class for MergeEngine failures."""


class InvalidConfiguration(MergeError):
    """Invalid table, constraint, cut, or identity configuration."""


class UnsupportedSemantics(MergeError):
    """Requested profile, legality, or decision capability is unsupported."""


class DecisionError(MergeError):
    """Dropout decision source exhausted or failed."""


class InvariantViolation(MergeError):
    """Internal adjacency / partition corruption."""


# ---------------------------------------------------------------------------
# Contract types
# ---------------------------------------------------------------------------


class SemanticProfile(str, Enum):
    """Supported merge-execution semantic profiles."""

    SUPER_BPE_PASS_V1 = "SuperBpePassV1"


PieceKey = str
TokenId = int
LeafIndex = int
LeafBoundary = int  # cut between leaf i-1 and leaf i; value is the right leaf index
VocabularyIdentity = object


@dataclass(frozen=True)
class Atom:
    """One input leaf. ``piece`` is an exact string key; leaf index is its position."""

    piece: PieceKey


@dataclass(frozen=True)
class MergeRule:
    """Resolved merge for an adjacent pair."""

    result: PieceKey
    result_id: TokenId
    learned_rank: Optional[int] = None


@dataclass(frozen=True)
class Group:
    """One output group covering ``leaves = [start, end)`` of the original input."""

    leaves: Tuple[LeafIndex, LeafIndex]
    piece: PieceKey
    merged_id: Optional[TokenId]

    @property
    def start(self) -> LeafIndex:
        return self.leaves[0]

    @property
    def end(self) -> LeafIndex:
        return self.leaves[1]


@dataclass(frozen=True)
class MergePlan:
    """Provenance returned by MergeEngine.apply."""

    groups: Tuple[Group, ...]
    applied_merges: int
    decisions_consumed: int


class MergeTableView(Protocol):
    """Immutable pair -> rule resolver for one vocabulary identity."""

    @property
    def vocabulary_identity(self) -> VocabularyIdentity: ...

    def resolve(self, left: PieceKey, right: PieceKey) -> Optional[MergeRule]: ...


class ImmutableLegalityView(Protocol):
    """Pure supplied legality predicate. Must not consult RNG or mutate state."""

    def allows(
        self,
        left: PieceKey,
        right: PieceKey,
        rule: MergeRule,
        leaf_range: Tuple[LeafIndex, LeafIndex],
    ) -> bool: ...


class DropoutDecisions(Protocol):
    """Caller-owned dropout decision cursor."""

    def drop_next(self) -> bool: ...


@dataclass(frozen=True)
class MergeConstraints:
    """Per-call constraints. Hard cuts and legality are experimental extensions."""

    semantic_profile: SemanticProfile
    vocabulary_identity: VocabularyIdentity
    hard_cuts: frozenset[LeafBoundary] = frozenset()
    legality: Optional[ImmutableLegalityView] = None


# ---------------------------------------------------------------------------
# Concrete adapters used by the reference oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MembershipMergeTable:
    """Legacy SuperBPE table: pair is eligible iff concatenation is in ``eligible``."""

    eligible: Mapping[PieceKey, TokenId]
    vocabulary_identity: VocabularyIdentity
    # Optional learned ranks retained as metadata only; SuperBpePassV1 ignores them.
    ranks: Mapping[Tuple[PieceKey, PieceKey], int] = field(default_factory=dict)

    def resolve(self, left: PieceKey, right: PieceKey) -> Optional[MergeRule]:
        result = left + right
        token_id = self.eligible.get(result)
        if token_id is None:
            return None
        rank = self.ranks.get((left, right))
        return MergeRule(result=result, result_id=token_id, learned_rank=rank)


@dataclass
class AlwaysAllowLegality:
    """Default production legality: every resolved rule is admitted."""

    def allows(
        self,
        left: PieceKey,
        right: PieceKey,
        rule: MergeRule,
        leaf_range: Tuple[LeafIndex, LeafIndex],
    ) -> bool:
        return True


@dataclass
class PredicateLegality:
    """Adapter around a pure caller-supplied predicate."""

    predicate: Callable[
        [PieceKey, PieceKey, MergeRule, Tuple[LeafIndex, LeafIndex]],
        bool,
    ]

    def allows(
        self,
        left: PieceKey,
        right: PieceKey,
        rule: MergeRule,
        leaf_range: Tuple[LeafIndex, LeafIndex],
    ) -> bool:
        return self.predicate(left, right, rule, leaf_range)


@dataclass
class BooleanTapeDecisions:
    """Reproducible dropout cursor over a finite Boolean tape.

    Exhaustion raises ``DecisionError`` and never draws from a fresh RNG.
    """

    tape: Sequence[bool]
    _cursor: int = 0

    def drop_next(self) -> bool:
        if self._cursor >= len(self.tape):
            raise DecisionError(f"dropout decision tape exhausted after {len(self.tape)} draws")
        value = self.tape[self._cursor]
        self._cursor += 1
        return value

    @property
    def consumed(self) -> int:
        return self._cursor


@dataclass
class PythonRandomDecisions:
    """Production-compatible dropout: ``random.random() < probability``."""

    probability: float
    draws: int = 0

    def drop_next(self) -> bool:
        self.draws += 1
        return random.random() < self.probability


# ---------------------------------------------------------------------------
# Reference engine
# ---------------------------------------------------------------------------


@dataclass
class _LiveGroup:
    """Mutable working group used only inside one ``apply`` call."""

    start: LeafIndex
    end: LeafIndex
    piece: PieceKey
    merged_id: Optional[TokenId]


class ReferenceMergeEngine:
    """Deliberately readable SuperBPE pass-loop oracle (``SuperBpePassV1``).

    Semantics match ``CustomTokenizer._apply_cross_word_merges`` /
    ``_apply_cross_word_merges_with_spans`` for production-compatible
    membership tables, empty hard cuts, always-allow legality, and Python RNG
    (or an equivalent Boolean tape). Experimental hard cuts / legality /
    decision sources are expressed in the same loop rather than silently
    ignored.

    Unsupported: any semantic profile other than ``SuperBpePassV1``. Rank
    values on rules are accepted as metadata and never used for scheduling.
    """

    SUPPORTED_PROFILES = frozenset({SemanticProfile.SUPER_BPE_PASS_V1})

    def apply(
        self,
        input: Sequence[Atom],
        table: MergeTableView,
        constraints: MergeConstraints,
        decisions: Optional[DropoutDecisions] = None,
    ) -> MergePlan:
        self._validate(input, table, constraints)
        if not input:
            return MergePlan(groups=(), applied_merges=0, decisions_consumed=0)

        current: List[_LiveGroup] = [
            _LiveGroup(start=i, end=i + 1, piece=atom.piece, merged_id=None) for i, atom in enumerate(input)
        ]
        # Blocked boundaries are keyed by the pair of *unchanged constituent
        # leaf ranges* (start, mid, end). Position shifts alone do not release
        # a block; a neighboring accepted merge that changes a constituent does.
        blocked: Set[Tuple[LeafIndex, LeafIndex, LeafIndex]] = set()
        decisions_consumed = 0
        initial_leaf_count = len(input)

        while True:
            nxt: List[_LiveGroup] = []
            accepted = 0
            cursor = 0
            while cursor < len(current):
                left = current[cursor]
                if cursor + 1 >= len(current):
                    nxt.append(left)
                    cursor += 1
                    continue

                right = current[cursor + 1]
                boundary_key = (left.start, left.end, right.end)
                rule = table.resolve(left.piece, right.piece)
                eligible = (
                    rule is not None
                    and boundary_key not in blocked
                    and left.end not in constraints.hard_cuts
                    and self._legality_allows(
                        constraints,
                        left.piece,
                        right.piece,
                        rule,
                        (left.start, right.end),
                    )
                )
                if not eligible:
                    nxt.append(left)
                    cursor += 1
                    continue

                assert rule is not None
                if decisions is not None:
                    try:
                        drop = decisions.drop_next()
                    except DecisionError:
                        raise
                    except Exception as exc:  # pragma: no cover - defensive
                        raise DecisionError(str(exc)) from exc
                    decisions_consumed += 1
                    if drop:
                        blocked.add(boundary_key)
                        nxt.append(left)
                        cursor += 1
                        continue

                # Accept: fuse left+right. Release this boundary and both
                # neighboring boundaries whose constituents just changed.
                merged = _LiveGroup(
                    start=left.start,
                    end=right.end,
                    piece=rule.result,
                    merged_id=rule.result_id,
                )
                if merged.piece != left.piece + right.piece:
                    raise InvariantViolation(
                        f"merge result {merged.piece!r} is not concatenation of {left.piece!r} + {right.piece!r}"
                    )
                self._release_neighbor_blocks(blocked, cursor, current)
                nxt.append(merged)
                cursor += 2
                accepted += 1

            if accepted == 0:
                groups = tuple(
                    Group(
                        leaves=(g.start, g.end),
                        piece=g.piece,
                        merged_id=g.merged_id,
                    )
                    for g in nxt
                )
                self._assert_partition(groups, initial_leaf_count)
                applied = initial_leaf_count - len(groups)
                return MergePlan(
                    groups=groups,
                    applied_merges=applied,
                    decisions_consumed=decisions_consumed,
                )
            current = nxt

    def _validate(
        self,
        input: Sequence[Atom],
        table: MergeTableView,
        constraints: MergeConstraints,
    ) -> None:
        if constraints.semantic_profile not in self.SUPPORTED_PROFILES:
            raise UnsupportedSemantics(
                f"unsupported semantic profile {constraints.semantic_profile!r}; "
                f"supported={sorted(p.value for p in self.SUPPORTED_PROFILES)}"
            )
        if constraints.vocabulary_identity is not table.vocabulary_identity:
            raise InvalidConfiguration("constraints.vocabulary_identity does not match table.vocabulary_identity")
        n = len(input)
        for cut in constraints.hard_cuts:
            if not isinstance(cut, int) or isinstance(cut, bool) or cut < 1 or cut >= n:
                raise InvalidConfiguration(f"hard cut {cut!r} is out of range for input length {n}")
        for atom in input:
            if not isinstance(atom.piece, str):
                raise InvalidConfiguration(f"Atom.piece must be a str PieceKey, got {type(atom.piece)!r}")

    @staticmethod
    def _legality_allows(
        constraints: MergeConstraints,
        left: PieceKey,
        right: PieceKey,
        rule: MergeRule,
        leaf_range: Tuple[LeafIndex, LeafIndex],
    ) -> bool:
        legality = constraints.legality
        if legality is None:
            return True
        return legality.allows(left, right, rule, leaf_range)

    @staticmethod
    def _release_neighbor_blocks(
        blocked: Set[Tuple[LeafIndex, LeafIndex, LeafIndex]],
        cursor: int,
        current: Sequence[_LiveGroup],
    ) -> None:
        """Drop blocks whose constituents changed because ``current[cursor]`` fused."""
        left = current[cursor]
        right = current[cursor + 1]
        # Shared boundary of the fused pair.
        blocked.discard((left.start, left.end, right.end))
        # Left neighbor boundary (prev, left) — left constituent changes.
        if cursor > 0:
            prev = current[cursor - 1]
            blocked.discard((prev.start, prev.end, left.end))
        # Right neighbor boundary (right, next) — right constituent changes.
        if cursor + 2 < len(current):
            nxt = current[cursor + 2]
            blocked.discard((right.start, right.end, nxt.end))

    @staticmethod
    def _assert_partition(groups: Sequence[Group], leaf_count: int) -> None:
        if leaf_count == 0:
            if groups:
                raise InvariantViolation("empty input produced non-empty groups")
            return
        cursor = 0
        for group in groups:
            start, end = group.leaves
            if start != cursor or end <= start:
                raise InvariantViolation(
                    f"groups do not form a contiguous partition: expected start {cursor}, got [{start}, {end})"
                )
            cursor = end
        if cursor != leaf_count:
            raise InvariantViolation(f"groups cover [0, {cursor}) but input has {leaf_count} leaves")


# ---------------------------------------------------------------------------
# Production adapters / projection helpers
# ---------------------------------------------------------------------------


def cross_word_membership_table(
    tokenizer: CustomTokenizer,
) -> MembershipMergeTable:
    """Build a legacy membership table from the tokenizer's cross-word vocab set."""
    eligible = {
        piece: tokenizer.model.token_to_id[piece]
        for piece in tokenizer._cross_word_tokens()
        if piece in tokenizer.model.token_to_id
    }
    # Identity is the frozenset of eligible pieces (stable for a model snapshot).
    identity = frozenset(eligible.items())
    return MembershipMergeTable(eligible=eligible, vocabulary_identity=identity)


def production_constraints(table: MembershipMergeTable) -> MergeConstraints:
    """Production-compatible constraints: SuperBpePassV1, no cuts, always-allow."""
    return MergeConstraints(
        semantic_profile=SemanticProfile.SUPER_BPE_PASS_V1,
        vocabulary_identity=table.vocabulary_identity,
        hard_cuts=frozenset(),
        legality=AlwaysAllowLegality(),
    )


def atoms_from_pieces(pieces: Sequence[str]) -> Tuple[Atom, ...]:
    return tuple(Atom(piece=p) for p in pieces)


def plan_to_pieces(plan: MergePlan) -> List[str]:
    return [group.piece for group in plan.groups]


def plan_to_tokens(plan: MergePlan, leaves: Sequence[Token]) -> List[Token]:
    """Project a plan onto original ``Token`` payloads (IDs + raw spans).

    Singleton groups retain the original leaf Token fields. Merged groups use
    the rule result ID and ``(leaves[start].raw_span[0], leaves[end-1].raw_span[1])``.
    Synthetic leaf ranges on the plan must never be treated as text offsets.
    """
    out: List[Token] = []
    for group in plan.groups:
        start, end = group.leaves
        if end - start == 1:
            out.append(leaves[start])
            continue
        if group.merged_id is None:
            raise InvariantViolation(f"merged group [{start}, {end}) is missing merged_id")
        out.append(
            Token(
                text=group.piece,
                id=group.merged_id,
                raw_span=(leaves[start].raw_span[0], leaves[end - 1].raw_span[1]),
            )
        )
    return out


def apply_reference_to_pieces(
    engine: ReferenceMergeEngine,
    pieces: Sequence[str],
    table: MergeTableView,
    constraints: MergeConstraints,
    decisions: Optional[DropoutDecisions] = None,
) -> Tuple[List[str], MergePlan]:
    """Convenience: atoms -> plan -> pieces."""
    plan = engine.apply(atoms_from_pieces(pieces), table, constraints, decisions)
    return plan_to_pieces(plan), plan


def apply_reference_to_tokens(
    engine: ReferenceMergeEngine,
    tokens: Sequence[Token],
    table: MergeTableView,
    constraints: MergeConstraints,
    decisions: Optional[DropoutDecisions] = None,
) -> Tuple[List[Token], MergePlan]:
    """Convenience: Token leaves -> plan -> Token projection."""
    pieces = [t.text for t in tokens]
    plan = engine.apply(atoms_from_pieces(pieces), table, constraints, decisions)
    return plan_to_tokens(plan, tokens), plan


def differential_against_production(
    tokenizer: CustomTokenizer,
    pieces: Sequence[str],
    dropout_prob: float = 0.0,
    *,
    decision_tape: Optional[Sequence[bool]] = None,
) -> Dict[str, object]:
    """Compare the reference oracle to both unmodified production helpers.

    When ``decision_tape`` is provided, both engines consume the same Boolean
    decisions (reference via ``BooleanTapeDecisions``; production via patched
    ``random.random`` values derived from the tape). When omitted and
    ``dropout_prob == 0``, no RNG is used. Non-zero dropout without a tape
    uses independent Python RNG streams and is unsuitable for strict parity.

    Returns a dict with pieces, span projections, merge/decision counts, and
    a ``match`` flag.
    """
    from unittest.mock import patch

    engine = ReferenceMergeEngine()
    table = cross_word_membership_table(tokenizer)
    constraints = production_constraints(table)

    if not table.eligible:
        # Production helpers short-circuit; reference also returns unchanged.
        leaves = [Token(p, tokenizer.model.token_to_id.get(p, -1), (i, i + 1)) for i, p in enumerate(pieces)]
        return {
            "production_pieces": list(pieces),
            "reference_pieces": list(pieces),
            "production_spans": [(t.raw_span, t.id, t.text) for t in leaves],
            "reference_spans": [(t.raw_span, t.id, t.text) for t in leaves],
            "decisions_consumed": 0,
            "applied_merges": 0,
            "match": True,
        }

    if dropout_prob > 0.0 and decision_tape is None:
        raise InvalidConfiguration("differential_against_production requires decision_tape when dropout_prob > 0")

    if decision_tape is None:
        ref_decisions: Optional[DropoutDecisions] = None
        prod_pieces = tokenizer._apply_cross_word_merges(list(pieces), dropout_prob=0.0)
        leaves = [Token(p, tokenizer.model.token_to_id.get(p, i), (i, i + 1)) for i, p in enumerate(pieces)]
        prod_tokens = tokenizer._apply_cross_word_merges_with_spans(leaves, dropout_prob=0.0)
        ref_pieces, plan = apply_reference_to_pieces(engine, pieces, table, constraints, ref_decisions)
        ref_tokens, _ = apply_reference_to_tokens(engine, leaves, table, constraints, None)
    else:
        # Map True(drop)->0.0 and False(keep)->0.99 so production's
        # ``random.random() < p`` with p=0.5 matches the Boolean tape.
        draws = [0.0 if drop else 0.99 for drop in decision_tape]
        with patch("random.random", side_effect=list(draws)):
            prod_pieces = tokenizer._apply_cross_word_merges(list(pieces), dropout_prob=dropout_prob)
        leaves = [Token(p, tokenizer.model.token_to_id.get(p, i), (i, i + 1)) for i, p in enumerate(pieces)]
        with patch("random.random", side_effect=list(draws)):
            prod_tokens = tokenizer._apply_cross_word_merges_with_spans(list(leaves), dropout_prob=dropout_prob)
        tape_a = BooleanTapeDecisions(list(decision_tape))
        ref_pieces, plan = apply_reference_to_pieces(engine, pieces, table, constraints, tape_a)
        tape_b = BooleanTapeDecisions(list(decision_tape))
        ref_tokens, _ = apply_reference_to_tokens(engine, leaves, table, constraints, tape_b)

    match = (
        ref_pieces == prod_pieces
        and [t.text for t in ref_tokens] == [t.text for t in prod_tokens]
        and [t.id for t in ref_tokens] == [t.id for t in prod_tokens]
        and [t.raw_span for t in ref_tokens] == [t.raw_span for t in prod_tokens]
    )
    return {
        "production_pieces": prod_pieces,
        "reference_pieces": ref_pieces,
        "production_spans": [(t.raw_span, t.id, t.text) for t in prod_tokens],
        "reference_spans": [(t.raw_span, t.id, t.text) for t in ref_tokens],
        "decisions_consumed": plan.decisions_consumed,
        "applied_merges": plan.applied_merges,
        "match": match,
    }


def document_semantic_invariants() -> str:
    """Human-readable summary of SuperBpePassV1 invariants for #108 docs/tests."""
    return """
SuperBpePassV1 semantic invariants (ReferenceMergeEngine)
=========================================================
1. Pair eligibility is concatenated-string membership in the supplied table.
   Learning history / CEM ranks do not schedule merges.
2. Each pass scans left-to-right and accepts non-overlapping pairs. A newly
   merged group cannot merge again until the next pass.
3. Overlapping competitors resolve to the leftmost eligible pair.
4. Passes repeat only while the preceding pass accepted a merge.
5. Positive dropout draws once per visited eligible unblocked pair. A dropped
   boundary stays blocked while both constituents are unchanged; a neighboring
   accepted merge that changes a constituent releases that block.
6. Zero dropout / decisions=None consumes no draws.
7. Hard cuts and legality are evaluated before any dropout decision; rejected
   pairs consume no decision.
8. Groups partition [0, len(input)); applied_merges == len(input) - len(groups).
9. Merged Token projection uses (first.raw_span[0], last.raw_span[1]) and the
   rule result_id; singletons retain original payload fields.
10. Unsupported profiles fail with UnsupportedSemantics; invalid cuts/identity
    fail with InvalidConfiguration; tape exhaustion raises DecisionError.
11. Special / byte / empty pieces are not filtered at runtime: membership alone
    decides. Leading-only metaspace markers are excluded by the *table builder*
    (_cross_word_tokens), not by the engine loop.
""".strip()
