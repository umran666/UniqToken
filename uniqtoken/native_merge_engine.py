"""Internal native merge prototype; no default selector or public API export.

Only compiled membership tables, hard cuts and finite Boolean decision tapes
are supported. Unsupported execution fails explicitly, without a Python fallback.
Source payloads remain with the caller and existing plan projection helpers.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Optional, Sequence

from .merge_engine import (
    AlwaysAllowLegality,
    Atom,
    BooleanTapeDecisions,
    DecisionError,
    DropoutDecisions,
    Group,
    InvalidConfiguration,
    InvariantViolation,
    MembershipMergeTable,
    MergeConstraints,
    MergePlan,
    MergeTableView,
    SemanticProfile,
    UnsupportedSemantics,
)

_METHODS = {
    (MembershipMergeTable, "resolve"): MembershipMergeTable.resolve,
    (AlwaysAllowLegality, "allows"): AlwaysAllowLegality.allows,
    (BooleanTapeDecisions, "drop_next"): BooleanTapeDecisions.drop_next,
}


def _utf8(piece: str) -> None:
    if type(piece) is not str:
        raise InvalidConfiguration("native pieces must be exact strings")
    try:
        piece.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UnsupportedSemantics("native merge prototype requires UTF-8-representable pieces") from exc


def _recognized(value: object, implementation: type, method: str) -> bool:
    return (
        type(value) is implementation
        and method not in vars(value)
        and getattr(implementation, method) is _METHODS[(implementation, method)]
    )


class NativeFastMergeEngine:
    """Experimental Rust SuperBpePassV1 engine with explicit capability limits."""

    @property
    def name(self) -> str:
        return "native-prototype"

    def apply(
        self,
        input: Sequence[Atom],
        table: MergeTableView,
        constraints: MergeConstraints,
        decisions: Optional[DropoutDecisions] = None,
    ) -> MergePlan:
        if constraints.semantic_profile != SemanticProfile.SUPER_BPE_PASS_V1:
            raise UnsupportedSemantics("native prototype supports only SuperBpePassV1")
        if not _recognized(table, MembershipMergeTable, "resolve"):
            raise UnsupportedSemantics("native prototype requires an unmodified MembershipMergeTable")
        assert isinstance(table, MembershipMergeTable)
        for mapping in (table.eligible, table.ranks):
            if not isinstance(mapping, Mapping):
                raise InvalidConfiguration("membership and ranks must be mappings")
            if type(mapping) is not dict:
                raise UnsupportedSemantics("native prototype requires plain dict membership and rank metadata")
        if constraints.vocabulary_identity is not table.vocabulary_identity:
            raise InvalidConfiguration("constraints.vocabulary_identity does not match table.vocabulary_identity")
        if constraints.legality is not None and not _recognized(constraints.legality, AlwaysAllowLegality, "allows"):
            raise UnsupportedSemantics("native prototype supports only absent or always-allow legality")
        if type(constraints.hard_cuts) is not frozenset:
            raise UnsupportedSemantics("native prototype requires plain frozenset hard cuts")
        for cut in constraints.hard_cuts:
            if type(cut) is not int or not 0 < cut < len(input):
                raise InvalidConfiguration(f"hard cut {cut!r} is out of range for input length {len(input)}")
        pieces = [atom.piece for atom in input]
        for piece in pieces:
            _utf8(piece)
        eligible = dict(table.eligible)
        for piece, token_id in eligible.items():
            _utf8(piece)
            if type(token_id) is not int or token_id < 0:
                raise InvalidConfiguration("native membership IDs must be nonnegative integers")
            if token_id > 2**64 - 1:
                raise UnsupportedSemantics("native membership IDs must fit u64")
        # Ranks are metadata only, never queue priorities.
        for pair, rank in table.ranks.items():
            if not isinstance(pair, tuple) or len(pair) != 2 or type(rank) is not int or rank < 0:
                raise InvalidConfiguration("ranks must map string pairs to nonnegative integers")
            for piece in pair:
                _utf8(piece)
        tape = None
        if decisions is not None:
            if not _recognized(decisions, BooleanTapeDecisions, "drop_next"):
                raise UnsupportedSemantics("native prototype requires an unmodified finite Boolean tape")
            assert isinstance(decisions, BooleanTapeDecisions)
            if type(decisions.tape) not in (list, tuple):
                raise UnsupportedSemantics("native prototype requires a plain list or tuple decision tape")
            cursor = decisions.consumed
            if type(cursor) is not int or not 0 <= cursor <= len(decisions.tape):
                raise InvalidConfiguration("invalid Boolean tape cursor")
            if any(type(value) is not bool for value in decisions.tape):
                raise InvalidConfiguration("native decision tape must contain only booleans")
            tape = list(decisions.tape[cursor:])
        try:
            native = importlib.import_module("uniqtoken_core.uniqtoken_core")
        except ImportError as exc:
            raise UnsupportedSemantics("fresh native extension with merge prototype is required") from exc
        execute = getattr(native, "_experimental_merge_plan", None)
        if not callable(execute):
            raise UnsupportedSemantics("native extension does not contain the merge prototype; rebuild it")
        groups, applied, consumed, error = execute(pieces, eligible, sorted(constraints.hard_cuts), tape)
        if decisions is not None:
            assert isinstance(decisions, BooleanTapeDecisions)
            decisions._cursor += consumed
        if error == "DecisionError":
            assert isinstance(decisions, BooleanTapeDecisions)
            raise DecisionError(f"dropout decision tape exhausted after {len(decisions.tape)} draws")
        if error == "InvalidConfiguration":
            raise InvalidConfiguration("invalid native merge configuration")
        if error is not None:
            raise InvariantViolation(f"native merge failure: {error}")
        return MergePlan(
            tuple(Group((start, end), piece, token_id) for start, end, piece, token_id in groups), applied, consumed
        )
