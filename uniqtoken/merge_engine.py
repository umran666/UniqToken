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

import heapq
import os
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
    Union,
    runtime_checkable,
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


@runtime_checkable
class MergeEngine(Protocol):
    """Abstract generic boundary for merge-execution engines.

    Both ReferenceMergeEngine (the correctness oracle) and FastMergeEngine
    (the high-performance incremental implementation) conform to this contract.
    """

    @property
    def name(self) -> str: ...

    def apply(
        self,
        input: Sequence[Atom],
        table: MergeTableView,
        constraints: MergeConstraints,
        decisions: Optional[DropoutDecisions] = None,
    ) -> MergePlan: ...


# ---------------------------------------------------------------------------
# Concrete adapters used by the reference oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MembershipMergeTable:
    """Legacy SuperBPE table: pair is eligible iff concatenation is in ``eligible``.

    A pair index over every (left, right) split of each eligible result is
    built once in ``__post_init__`` so :meth:`resolve` costs two dict
    lookups instead of a string concatenation per visited pair per pass
    (#113). ``eligible`` / ``ranks`` are treated as immutable snapshots;
    mutating them after construction leaves the index stale.
    """

    eligible: Mapping[PieceKey, TokenId]
    vocabulary_identity: VocabularyIdentity
    # Optional learned ranks retained as metadata only; SuperBpePassV1 ignores them.
    ranks: Mapping[Tuple[PieceKey, PieceKey], int] = field(default_factory=dict)
    _pair_rules: Dict[PieceKey, Dict[PieceKey, MergeRule]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        index: Dict[PieceKey, Dict[PieceKey, MergeRule]] = {}
        get_rank = self.ranks.get
        for result, token_id in self.eligible.items():
            # cut in range(len(result) + 1) also covers the empty-piece
            # splits ("" + result and result + ""), which are real pairs.
            for cut in range(len(result) + 1):
                left, right = result[:cut], result[cut:]
                right_rules = index.get(left)
                if right_rules is None:
                    right_rules = {}
                    index[left] = right_rules
                right_rules[right] = MergeRule(
                    result=result,
                    result_id=token_id,
                    learned_rank=get_rank((left, right)),
                )
        object.__setattr__(self, "_pair_rules", index)

    def resolve(self, left: PieceKey, right: PieceKey) -> Optional[MergeRule]:
        right_rules = self._pair_rules.get(left)
        if right_rules is None:
            return None
        return right_rules.get(right)


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

    @property
    def name(self) -> str:
        return "reference"

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


class FastMergeEngine:
    """High-performance incremental SuperBPE merge engine (``SuperBpePassV1``).

    Implements the generic ``MergeEngine`` contract using an in-place doubly-linked
    flat array sequence without per-node object allocations or heap overhead.
    Avoids full-sequence re-scans and allocations across merge passes.

    Preserves exact SuperBpePassV1 semantics:
    - Deterministic pass barriers: newly formed merges cannot merge again in
      the same pass.
    - Leftmost overlap resolution.
    - Exact dropout draw consumption and boundary block lifetimes.
    - Boundary constraints (hard cuts) and pure supplied legality predicates.
    - Fails loudly on unsupported semantics rather than falling back silently.
    """

    SUPPORTED_PROFILES = frozenset({SemanticProfile.SUPER_BPE_PASS_V1})

    @property
    def name(self) -> str:
        return "fast"

    def apply(
        self,
        input: Sequence[Atom],
        table: MergeTableView,
        constraints: MergeConstraints,
        decisions: Optional[DropoutDecisions] = None,
    ) -> MergePlan:
        self._validate(input, table, constraints)
        n = len(input)
        if n == 0:
            return MergePlan(groups=(), applied_merges=0, decisions_consumed=0)
        if n == 1:
            atom = input[0]
            group = Group(leaves=(0, 1), piece=atom.piece, merged_id=None)
            return MergePlan(groups=(group,), applied_merges=0, decisions_consumed=0)

        pieces: List[str] = [atom.piece for atom in input]
        starts: List[int] = list(range(n))
        ends: List[int] = list(range(1, n + 1))
        next_idx: List[int] = list(range(1, n + 1))
        next_idx[-1] = -1
        prev_idx: List[int] = list(range(-1, n - 1))
        merged_ids: List[Optional[TokenId]] = [None] * n

        hard_cuts = constraints.hard_cuts
        has_cuts = bool(hard_cuts)
        legality = constraints.legality
        if type(legality) is AlwaysAllowLegality:
            # Production default: admitting every resolved rule is equivalent
            # to supplying no legality predicate, so the engine may take the
            # fast path instead of paying a per-pair predicate call (#113).
            legality = None
        has_legality = legality is not None
        resolve = table.resolve
        # rule.result is the concatenation of the pair by construction for the
        # built-in membership table; custom tables keep the defensive check.
        check_rule_concat = type(table) is not MembershipMergeTable

        blocked: Set[Tuple[LeafIndex, LeafIndex, LeafIndex]] = set()
        decisions_consumed = 0
        total_applied = 0
        head = 0

        # Fast path check for common case without cuts, legality predicates, or dropout
        is_clean_path = not has_cuts and not has_legality and decisions is None

        while True:
            accepted = 0
            u = head
            if is_clean_path and not blocked:
                while u != -1:
                    v = next_idx[u]
                    if v == -1:
                        break
                    rule = resolve(pieces[u], pieces[v])
                    if rule is None:
                        u = v
                        continue
                    if check_rule_concat and rule.result != pieces[u] + pieces[v]:
                        raise InvariantViolation(
                            f"merge result {rule.result!r} is not concatenation of {pieces[u]!r} + {pieces[v]!r}"
                        )
                    pieces[u] = rule.result
                    ends[u] = ends[v]
                    merged_ids[u] = rule.result_id
                    next_v = next_idx[v]
                    next_idx[u] = next_v
                    if next_v != -1:
                        prev_idx[next_v] = u
                    accepted += 1
                    total_applied += 1
                    u = next_v
            else:
                while u != -1:
                    v = next_idx[u]
                    if v == -1:
                        break
                    u_end = ends[u]
                    if has_cuts and u_end in hard_cuts:
                        u = v
                        continue
                    u_start = starts[u]
                    v_end = ends[v]
                    b_key = (u_start, u_end, v_end)
                    if blocked and b_key in blocked:
                        u = v
                        continue
                    rule = resolve(pieces[u], pieces[v])
                    if rule is None:
                        u = v
                        continue
                    if legality is not None and not legality.allows(pieces[u], pieces[v], rule, (u_start, v_end)):
                        u = v
                        continue
                    if decisions is not None:
                        try:
                            drop = decisions.drop_next()
                        except DecisionError:
                            raise
                        except Exception as exc:  # pragma: no cover - defensive
                            raise DecisionError(str(exc)) from exc
                        decisions_consumed += 1
                        if drop:
                            blocked.add(b_key)
                            u = v
                            continue

                    if check_rule_concat and rule.result != pieces[u] + pieces[v]:
                        raise InvariantViolation(
                            f"merge result {rule.result!r} is not concatenation of {pieces[u]!r} + {pieces[v]!r}"
                        )
                    pieces[u] = rule.result
                    ends[u] = v_end
                    merged_ids[u] = rule.result_id
                    next_v = next_idx[v]
                    next_idx[u] = next_v
                    if next_v != -1:
                        prev_idx[next_v] = u
                    if blocked:
                        blocked.discard(b_key)
                        p_u = prev_idx[u]
                        if p_u != -1:
                            blocked.discard((starts[p_u], ends[p_u], u_end))
                        if next_v != -1:
                            blocked.discard((u_end, ends[v], ends[next_v]))
                    accepted += 1
                    total_applied += 1
                    u = next_v

            if accepted == 0:
                break

        out_groups: List[Group] = []
        curr = head
        while curr != -1:
            out_groups.append(
                Group(
                    leaves=(starts[curr], ends[curr]),
                    piece=pieces[curr],
                    merged_id=merged_ids[curr],
                )
            )
            curr = next_idx[curr]

        groups = tuple(out_groups)
        self._assert_partition(groups, n)
        return MergePlan(
            groups=groups,
            applied_merges=total_applied,
            decisions_consumed=decisions_consumed,
        )

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


DISABLE_ENGINE_ALIASES = frozenset({"default", "production", "none", "off"})
FAST_ENGINE_ALIASES = frozenset({"fast", "fast_merge", "fast_merge_engine"})
REFERENCE_ENGINE_ALIASES = frozenset({"reference", "oracle"})


def get_merge_engine(name_or_instance: Union[str, MergeEngine, None] = None) -> Optional[MergeEngine]:
    """Resolve a MergeEngine instance by name or pass through an existing instance.

    Supported names:
    - "reference" (or "oracle"): ReferenceMergeEngine
    - "fast" (or "fast_merge", "fast_merge_engine"): FastMergeEngine
    - "default", "production", "none", "off": None (uses default inlined production path)

    If name_or_instance is None, checks the UNIQTOKEN_MERGE_ENGINE environment variable.
    If UNIQTOKEN_MERGE_ENGINE is unset, empty, or a disable alias, returns None.
    Unknown engine names or invalid UNIQTOKEN_MERGE_ENGINE values raise InvalidConfiguration.
    """
    if name_or_instance is None:
        env_name = os.environ.get("UNIQTOKEN_MERGE_ENGINE", "").strip().lower()
        if not env_name or env_name in DISABLE_ENGINE_ALIASES:
            return None
        if env_name in FAST_ENGINE_ALIASES:
            return FastMergeEngine()
        if env_name in REFERENCE_ENGINE_ALIASES:
            return ReferenceMergeEngine()
        raise InvalidConfiguration(
            f"Unknown engine name in UNIQTOKEN_MERGE_ENGINE: {env_name!r}; expected 'fast', 'reference', or 'default'"
        )

    if isinstance(name_or_instance, str):
        normalized = name_or_instance.strip().lower()
        if normalized in DISABLE_ENGINE_ALIASES:
            return None
        if normalized in FAST_ENGINE_ALIASES:
            return FastMergeEngine()
        if normalized in REFERENCE_ENGINE_ALIASES:
            return ReferenceMergeEngine()
        raise InvalidConfiguration(
            f"Unknown merge engine name: {name_or_instance!r}; expected 'fast', 'reference', or 'default'"
        )

    if hasattr(name_or_instance, "apply") and callable(name_or_instance.apply):
        return name_or_instance

    raise InvalidConfiguration(f"Expected a str name, MergeEngine instance, or None, got {type(name_or_instance)!r}")


def apply_engine_to_pieces(
    engine: MergeEngine,
    pieces: Sequence[str],
    table: MergeTableView,
    constraints: MergeConstraints,
    decisions: Optional[DropoutDecisions] = None,
) -> Tuple[List[str], MergePlan]:
    """Convenience: atoms -> plan -> pieces using any MergeEngine."""
    plan = engine.apply(atoms_from_pieces(pieces), table, constraints, decisions)
    return plan_to_pieces(plan), plan


def apply_engine_to_tokens(
    engine: MergeEngine,
    tokens: Sequence[Token],
    table: MergeTableView,
    constraints: MergeConstraints,
    decisions: Optional[DropoutDecisions] = None,
) -> Tuple[List[Token], MergePlan]:
    """Convenience: Token leaves -> plan -> Token projection using any MergeEngine."""
    pieces = [t.text for t in tokens]
    plan = engine.apply(atoms_from_pieces(pieces), table, constraints, decisions)
    return plan_to_tokens(plan, tokens), plan


def apply_reference_to_pieces(
    engine: ReferenceMergeEngine,
    pieces: Sequence[str],
    table: MergeTableView,
    constraints: MergeConstraints,
    decisions: Optional[DropoutDecisions] = None,
) -> Tuple[List[str], MergePlan]:
    """Convenience: atoms -> plan -> pieces."""
    return apply_engine_to_pieces(engine, pieces, table, constraints, decisions)


def apply_reference_to_tokens(
    engine: ReferenceMergeEngine,
    tokens: Sequence[Token],
    table: MergeTableView,
    constraints: MergeConstraints,
    decisions: Optional[DropoutDecisions] = None,
) -> Tuple[List[Token], MergePlan]:
    """Convenience: Token leaves -> plan -> Token projection."""
    return apply_engine_to_tokens(engine, tokens, table, constraints, decisions)


def differential_against_production(
    tokenizer: CustomTokenizer,
    pieces: Sequence[str],
    dropout_prob: float = 0.0,
    *,
    decision_tape: Optional[Sequence[bool]] = None,
    engine: Optional[MergeEngine] = None,
) -> Dict[str, object]:
    """Compare a MergeEngine (default ReferenceMergeEngine) to both unmodified production helpers.

    When ``decision_tape`` is provided, both engines consume the same Boolean
    decisions (reference via ``BooleanTapeDecisions``; production via patched
    ``random.random`` values derived from the tape). When omitted and
    ``dropout_prob == 0``, no RNG is used. Non-zero dropout without a tape
    uses independent Python RNG streams and is unsuitable for strict parity.

    Returns a dict with pieces, span projections, merge/decision counts, and
    a ``match`` flag.
    """
    from unittest.mock import patch

    eval_engine: MergeEngine = engine if engine is not None else ReferenceMergeEngine()
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

    # Isolate production baseline execution from any configured experimental merge engine
    old_engine = getattr(tokenizer, "_resolved_merge_engine", None)
    tokenizer._resolved_merge_engine = None
    try:
        if decision_tape is None:
            ref_decisions: Optional[DropoutDecisions] = None
            prod_pieces = tokenizer._apply_cross_word_merges(list(pieces), dropout_prob=0.0)
            leaves = [Token(p, tokenizer.model.token_to_id.get(p, i), (i, i + 1)) for i, p in enumerate(pieces)]
            prod_tokens = tokenizer._apply_cross_word_merges_with_spans(leaves, dropout_prob=0.0)
            ref_pieces, plan = apply_engine_to_pieces(eval_engine, pieces, table, constraints, ref_decisions)
            ref_tokens, _ = apply_engine_to_tokens(eval_engine, leaves, table, constraints, None)
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
            ref_pieces, plan = apply_engine_to_pieces(eval_engine, pieces, table, constraints, tape_a)
            tape_b = BooleanTapeDecisions(list(decision_tape))
            ref_tokens, _ = apply_engine_to_tokens(eval_engine, leaves, table, constraints, tape_b)
    finally:
        tokenizer._resolved_merge_engine = old_engine

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


def differential_reference_vs_fast(
    pieces: Sequence[str],
    table: MergeTableView,
    constraints: MergeConstraints,
    decisions_factory: Optional[Callable[[], DropoutDecisions]] = None,
) -> Dict[str, object]:
    """Compare ReferenceMergeEngine and FastMergeEngine on exact same input."""
    ref_engine = ReferenceMergeEngine()
    fast_engine = FastMergeEngine()
    atoms = atoms_from_pieces(pieces)

    dec_ref = decisions_factory() if decisions_factory else None
    dec_fast = decisions_factory() if decisions_factory else None

    ref_plan = ref_engine.apply(atoms, table, constraints, dec_ref)
    fast_plan = fast_engine.apply(atoms, table, constraints, dec_fast)

    match = (
        plan_to_pieces(ref_plan) == plan_to_pieces(fast_plan)
        and ref_plan.applied_merges == fast_plan.applied_merges
        and ref_plan.decisions_consumed == fast_plan.decisions_consumed
        and [g.leaves for g in ref_plan.groups] == [g.leaves for g in fast_plan.groups]
        and [g.merged_id for g in ref_plan.groups] == [g.merged_id for g in fast_plan.groups]
    )
    return {
        "reference_pieces": plan_to_pieces(ref_plan),
        "fast_pieces": plan_to_pieces(fast_plan),
        "reference_applied": ref_plan.applied_merges,
        "fast_applied": fast_plan.applied_merges,
        "reference_decisions": ref_plan.decisions_consumed,
        "fast_decisions": fast_plan.decisions_consumed,
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
