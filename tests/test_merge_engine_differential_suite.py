"""Merge-engine differential verification suite (Issue #110).

Empirically verifies the semantic-equality contract between
:class:`ReferenceMergeEngine` (the correctness oracle) and
:class:`FastMergeEngine` (the incremental implementation) under the
``SuperBpePassV1`` profile, across every differential category required
by issue #110:

- deterministic fixtures;
- random token sequences and merge tables;
- duplicate/conflicting pair ranks (rank metadata must never schedule);
- boundary-constrained (hard cuts + legality) and cross-word merges;
- byte-fallback and special-token piece keys;
- empty inputs and repeated tokens;
- pathological long sequences;
- Unicode-derived token sequences;
- dropout/invalidation cases under reproducible random control
  (fixed-seed Boolean decision tapes at several densities).

For every supported case the suite compares the full equality contract
from #110: output pieces, group leaf ranges, merged token IDs, applied
merge count, decision-draw consumption, and output length. Error cases
compare exception classification (and tape consumption on decision
exhaustion) instead of plans.

Acceptance-criteria mapping:

- zero mismatches: every test below asserts exact plan equality;
- fixed and recorded random seeds: ``MASTER_SEED`` plus per-case seeds
  recorded in the artifact;
- machine-readable artifact: written to
  ``artifacts/merge_engine_differential_report.json`` with case counts,
  configuration, commit, platform, and input hashes;
- failure reduction: any mismatch is delta-debugged down to a minimal
  reproducible case before the assertion fires;
- platform / build coverage: both engines on this path are pure Python,
  so there are no release/debug build variants to matrix; the artifact
  records the platform and Python build instead. The native merge
  prototype (issue #109) is out of scope here;
- no performance conclusion is drawn from this correctness suite; all
  performance questions belong to the benchmark issues (#111, #113).

Run independently with::

    python -m unittest tests.test_merge_engine_differential_suite -v
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import random
import subprocess
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from uniqtoken.merge_engine import (
    Atom,
    BooleanTapeDecisions,
    DecisionError,
    FastMergeEngine,
    InvalidConfiguration,
    MembershipMergeTable,
    MergeConstraints,
    MergePlan,
    PredicateLegality,
    ReferenceMergeEngine,
    SemanticProfile,
    UnsupportedSemantics,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "artifacts" / "merge_engine_differential_report.json"

# Fixed and recorded master seed (issue #110 acceptance criterion). Every
# randomized case derives its own seed deterministically from this one.
MASTER_SEED = 20261024
CASES_PER_RANDOM_CATEGORY = 40
PATHOLOGICAL_LENGTH = 5000

SPECIAL_PIECES = ["<|endoftext|>", "<|fim_pad|>", "\x00", "\xff", "\u00e9"]
UNICODE_PIECES = [
    "cafe\u0301",
    "\u0928\u092e\u0938\u094d\u0924\u0947",
    "\u4e16\u754c",
    "\ud55c\uad6d\uc5b4",
    "\U0001f469\u200d\U0001f4bb",
    "\u0627\u0644\u0639\u0631\u0628\u064a\u0629",
]

# Case type: JSON-serializable descriptor of one differential case.
Case = Dict[str, Any]


# ---------------------------------------------------------------------------
# Case construction helpers
# ---------------------------------------------------------------------------


def _case(
    category: str,
    pieces: Sequence[str],
    eligible: Dict[str, int],
    *,
    seed: Optional[int] = None,
    ranks: Optional[Dict[Tuple[str, str], int]] = None,
    hard_cuts: Sequence[int] = (),
    legality: Optional[Dict[str, Any]] = None,
    tape: Optional[Sequence[bool]] = None,
    expected_error: Optional[str] = None,
    constraints_identity_mismatch: bool = False,
    bad_atom: bool = False,
    semantic_profile: Any = SemanticProfile.SUPER_BPE_PASS_V1,
) -> Case:
    return {
        "category": category,
        "seed": seed,
        "pieces": list(pieces),
        "eligible": dict(eligible),
        "ranks": {f"{l}\x00{r}": v for (l, r), v in (ranks or {}).items()},
        "hard_cuts": sorted(hard_cuts),
        "legality": legality,
        "tape": None if tape is None else list(tape),
        "expected_error": expected_error,
        "constraints_identity_mismatch": constraints_identity_mismatch,
        "bad_atom": bad_atom,
        "semantic_profile": (semantic_profile if isinstance(semantic_profile, str) else semantic_profile.value),
    }


def _build_fixtures(case: Case) -> Tuple[MembershipMergeTable, MergeConstraints, Tuple[Atom, ...]]:
    """Materialize table / constraints / atoms from a case descriptor."""
    ranks = {tuple(key.split("\x00", 1)): v for key, v in case["ranks"].items()}
    identity = frozenset(case["eligible"].items())
    table = MembershipMergeTable(
        eligible=dict(case["eligible"]),
        vocabulary_identity=identity,
        ranks=ranks,
    )
    legality: Optional[PredicateLegality] = None
    spec = case["legality"]
    if spec is not None:
        if spec["kind"] == "max_span":
            max_span = int(spec["max"])
            legality = PredicateLegality(
                lambda left, right, rule, leaf_range, _m=max_span: (leaf_range[1] - leaf_range[0]) <= _m
            )
        elif spec["kind"] == "forbid_left":
            forbidden = str(spec["piece"])
            legality = PredicateLegality(lambda left, right, rule, leaf_range, _p=forbidden: left != _p)
        else:  # pragma: no cover - generator only emits known kinds
            raise ValueError(f"unknown legality spec {spec!r}")
    constraints = MergeConstraints(
        semantic_profile=case["semantic_profile"],
        vocabulary_identity="mismatch" if case["constraints_identity_mismatch"] else identity,
        hard_cuts=frozenset(case["hard_cuts"]),
        legality=legality,
    )
    if case["bad_atom"]:
        atoms = (Atom(piece="a"), Atom(piece=123))  # type: ignore[list-item]
    else:
        atoms = tuple(Atom(piece=p) for p in case["pieces"])
    return table, constraints, atoms


# ---------------------------------------------------------------------------
# Execution + equality contract
# ---------------------------------------------------------------------------


def _run_one(
    engine: Any,
    case: Case,
    table: MembershipMergeTable,
    constraints: MergeConstraints,
    atoms: Tuple[Atom, ...],
) -> Tuple[Optional[MergePlan], Optional[type], Optional[int]]:
    """Run one engine on the case. Returns (plan | None, exc type | None, tape consumed)."""
    tape_decisions: Optional[BooleanTapeDecisions] = None
    if case["tape"] is not None:
        tape_decisions = BooleanTapeDecisions(list(case["tape"]))
    try:
        plan = engine.apply(atoms, table, constraints, tape_decisions)
        return plan, None, tape_decisions.consumed if tape_decisions is not None else None
    except Exception as exc:  # noqa: BLE001 - classification compared below
        if tape_decisions is not None and isinstance(exc, DecisionError):
            return None, type(exc), tape_decisions.consumed
        return None, type(exc), None


def _compare_case(case: Case) -> Tuple[bool, str]:
    """Run both engines and verify the #110 equality contract.

    Returns ``(match, reason)``; ``reason`` is empty when the case matches.
    """
    table, constraints, atoms = _build_fixtures(case)
    ref = ReferenceMergeEngine()
    fast = FastMergeEngine()

    ref_plan, ref_exc, ref_consumed = _run_one(ref, case, table, constraints, atoms)
    fast_plan, fast_exc, fast_consumed = _run_one(fast, case, table, constraints, atoms)

    expected = case["expected_error"]
    if expected is not None:
        if ref_exc is None or fast_exc is None:
            return False, f"expected {expected}, got ref_exc={ref_exc} fast_exc={fast_exc}"
        if ref_exc.__name__ != expected or fast_exc.__name__ != expected:
            return False, (
                f"error classification mismatch: expected {expected}, ref={ref_exc.__name__}, fast={fast_exc.__name__}"
            )
        if ref_consumed is not None and ref_consumed != fast_consumed:
            return False, f"tape consumption mismatch on error path: {ref_consumed} != {fast_consumed}"
        return True, ""

    if ref_exc is not None or fast_exc is not None:
        return False, f"unexpected error: ref={ref_exc} fast={fast_exc}"
    assert ref_plan is not None and fast_plan is not None

    if ref_plan.groups != fast_plan.groups:
        return False, (
            f"groups differ: ref={[(g.leaves, g.piece, g.merged_id) for g in ref_plan.groups][:8]}... "
            f"fast={[(g.leaves, g.piece, g.merged_id) for g in fast_plan.groups][:8]}..."
        )
    if ref_plan.applied_merges != fast_plan.applied_merges:
        return False, f"applied_merges differ: {ref_plan.applied_merges} != {fast_plan.applied_merges}"
    if ref_plan.decisions_consumed != fast_plan.decisions_consumed:
        return False, (f"decisions_consumed differ: {ref_plan.decisions_consumed} != {fast_plan.decisions_consumed}")
    if len(ref_plan.groups) != len(fast_plan.groups):  # pragma: no cover - implied by groups equality
        return False, "output lengths differ"
    return True, ""


# ---------------------------------------------------------------------------
# Failure reduction (issue #110: minimal reproducible case)
# ---------------------------------------------------------------------------


def _shrink_pieces(case: Case) -> List[str]:
    """Delta-debug the input pieces down to a minimal still-failing sequence.

    The table, cuts, legality spec and tape are kept as-is: candidates that
    fall out of range (e.g. a cut beyond the shortened input) now raise the
    same error class on *both* engines and therefore no longer count as
    failures, which automatically keeps the reduction sound.
    """
    pieces = list(case["pieces"])

    def still_failing(candidate: List[str]) -> bool:
        match, _ = _compare_case(dict(case, pieces=candidate))
        return not match

    chunk = max(1, len(pieces) // 2)
    while chunk >= 1:
        index = 0
        while index < len(pieces):
            removed = pieces[:index] + pieces[index + chunk :]
            if still_failing(removed):
                pieces = removed
            else:
                index += chunk
        if chunk == 1:
            break
        chunk = max(1, chunk // 2)
    return pieces


def _format_repro(case: Case, shrunk: Optional[List[str]] = None) -> str:
    shown = dict(case)
    if shrunk is not None:
        shown["pieces"] = shrunk
        shown["note"] = "pieces were reduced by delta debugging; all other fields are the original case"
    return json.dumps(shown, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Randomized category generators (all seeds derived from MASTER_SEED)
# ---------------------------------------------------------------------------


def _random_tape(rng: random.Random, length: int, density: float) -> List[bool]:
    return [rng.random() < density for _ in range(length)]


def _random_table(rng: random.Random, pieces: Sequence[str]) -> Dict[str, int]:
    candidates: List[str] = []
    for i in range(len(pieces) - 1):
        candidates.append(pieces[i] + pieces[i + 1])
    for i in range(len(pieces) - 2):
        candidates.append(pieces[i] + pieces[i + 1] + pieces[i + 2])
    eligible: Dict[str, int] = {}
    for result in dict.fromkeys(candidates):
        if rng.random() < 0.6:
            eligible[result] = rng.randrange(1, 1000)
    return eligible


def _gen_random_sequences(seed: int) -> List[Case]:
    rng = random.Random(seed)
    pool = ["a", "b", "ab", "\u2581b", "\u2581c", "cd"]
    cases: List[Case] = []
    for _ in range(CASES_PER_RANDOM_CATEGORY):
        n = rng.randint(0, 12)
        pieces = [rng.choice(pool) for _ in range(n)]
        eligible = _random_table(rng, pieces)
        tape = _random_tape(rng, 8 * n + 64, rng.choice([0.0, 0.3, 0.5])) if pieces and eligible else None
        cases.append(_case("random_sequences", pieces, eligible, seed=seed, tape=tape))
    return cases


def _gen_conflicting_ranks(seed: int) -> List[Case]:
    rng = random.Random(seed)
    cases: List[Case] = [
        # Same concatenation reachable via different splits with conflicting
        # rank metadata: SuperBpePassV1 must ignore ranks entirely.
        _case(
            "conflicting_ranks",
            ["a", "bc", "ab", "c"],
            {"abc": 7},
            ranks={("a", "bc"): 1, ("ab", "c"): 999},
            seed=seed,
        ),
    ]
    for _ in range(CASES_PER_RANDOM_CATEGORY):
        n = rng.randint(2, 10)
        pieces = [rng.choice(["a", "b", "c"]) for _ in range(n)]
        eligible = _random_table(rng, pieces)
        # Arbitrary, conflicting rank metadata over every adjacent pair.
        ranks = {(pieces[i], pieces[i + 1]): rng.randrange(0, 10_000) for i in range(n - 1)}
        cases.append(_case("conflicting_ranks", pieces, eligible, seed=seed, ranks=ranks))
    return cases


def _gen_boundary_constraints(seed: int) -> List[Case]:
    rng = random.Random(seed)
    pool = ["a", "b", "\u2581b", "\u2581c"]
    cases: List[Case] = []
    for _ in range(CASES_PER_RANDOM_CATEGORY):
        n = rng.randint(2, 14)
        pieces = [rng.choice(pool) for _ in range(n)]
        eligible = _random_table(rng, pieces)
        hard_cuts = [b for b in range(1, n) if rng.random() < 0.25]
        legality: Optional[Dict[str, Any]] = None
        if rng.random() < 0.5:
            legality = (
                {"kind": "max_span", "max": rng.randint(2, 4)}
                if rng.random() < 0.5
                else {"kind": "forbid_left", "piece": rng.choice(pool)}
            )
        tape = _random_tape(rng, 32 * n + 64, rng.choice([0.0, 0.3, 0.5]))
        cases.append(
            _case(
                "boundary_constraints",
                pieces,
                eligible,
                seed=seed,
                hard_cuts=hard_cuts,
                legality=legality,
                tape=tape,
            )
        )
    return cases


def _gen_byte_special(seed: int) -> List[Case]:
    rng = random.Random(seed)
    cases: List[Case] = []
    for _ in range(CASES_PER_RANDOM_CATEGORY):
        n = rng.randint(0, 10)
        pieces = [rng.choice(SPECIAL_PIECES) for _ in range(n)]
        eligible = _random_table(rng, pieces)
        tape = _random_tape(rng, 8 * n + 64, 0.3) if pieces and eligible else None
        cases.append(_case("byte_special_tokens", pieces, eligible, seed=seed, tape=tape))
    return cases


def _gen_empty_repeated(seed: int) -> List[Case]:
    rng = random.Random(seed)
    cases: List[Case] = [
        _case("empty_and_repeated", [], {}, seed=seed),
        _case("empty_and_repeated", ["a"], {"a": 1}, seed=seed),
        _case("empty_and_repeated", ["", "a", ""], {"a": 1, "": 2}, seed=seed),
    ]
    for _ in range(CASES_PER_RANDOM_CATEGORY):
        piece = rng.choice(["a", "\u2581b", "<|pad|>", "\u4e16"])
        n = rng.randint(2, 8)
        pieces = [piece] * n
        eligible: Dict[str, int] = {}
        # Dense self-concat chain: piece*2, piece*4, piece*8 all mergeable.
        power = 2
        while power <= n:
            eligible[piece * power] = rng.randrange(1, 1000)
            power *= 2
        tape = _random_tape(rng, 8 * n + 64, rng.choice([0.0, 0.5]))
        cases.append(_case("empty_and_repeated", pieces, eligible, seed=seed, tape=tape))
    return cases


def _gen_pathological_long(seed: int) -> List[Case]:
    rng = random.Random(seed)
    long_random = [rng.choice(["a", "b", "c", "d"]) for _ in range(2000)]
    dense_long = [rng.choice(["a", "b", "c", "d"]) for _ in range(2000)]
    return [
        # Dense chain: every adjacent pair merges, forcing pass-after-pass
        # re-scan of a collapsing sequence (log2(n) passes, n-1 merges).
        _case(
            "pathological_long",
            ["a"] * PATHOLOGICAL_LENGTH,
            {"aa": 1},
            seed=seed,
        ),
        _case(
            "pathological_long",
            long_random,
            {"ab": 1, "bc": 2, "cd": 3, "abc": 4, "bcd": 5, "cc": 6, "bb": 7},
            seed=seed,
        ),
        _case(
            "pathological_long",
            dense_long,
            {"aa": 1, "ab": 2, "bc": 3, "cd": 4, "da": 5, "dd": 6},
            hard_cuts=[500, 1000, 1500],
            seed=seed,
        ),
    ]


def _gen_unicode(seed: int) -> List[Case]:
    rng = random.Random(seed)
    cases: List[Case] = []
    for _ in range(CASES_PER_RANDOM_CATEGORY):
        n = rng.randint(0, 10)
        pieces = [rng.choice(UNICODE_PIECES) for _ in range(n)]
        eligible = _random_table(rng, pieces)
        tape = _random_tape(rng, 8 * n + 64, 0.3) if pieces and eligible else None
        cases.append(_case("unicode_sequences", pieces, eligible, seed=seed, tape=tape))
    return cases


def _gen_dropout_invalidation(seed: int) -> List[Case]:
    rng = random.Random(seed)
    cases: List[Case] = []
    for _ in range(CASES_PER_RANDOM_CATEGORY):
        n = rng.randint(4, 16)
        pieces = [rng.choice(["a", "b", "c", "\u2581d"]) for _ in range(n)]
        eligible = _random_table(rng, pieces)
        if not eligible:
            eligible[pieces[0] + pieces[1]] = 42
        # Dropout is where boundary-block release semantics live; sweep the
        # density range including the all-drop extremes.
        density = rng.choice([0.0, 0.2, 0.5, 0.9])
        tape = _random_tape(rng, 32 * n + 64, density)
        cases.append(_case("dropout_invalidation", pieces, eligible, seed=seed, tape=tape))
    return cases


def _gen_error_classification(seed: int) -> List[Case]:
    return [
        _case(
            "error_classification",
            ["a", "b"],
            {"ab": 1},
            semantic_profile="SuperBpePassV2",
            expected_error="UnsupportedSemantics",
            seed=seed,
        ),
        _case(
            "error_classification",
            ["a", "b"],
            {"ab": 1},
            constraints_identity_mismatch=True,
            expected_error="InvalidConfiguration",
            seed=seed,
        ),
        _case(
            "error_classification",
            ["a", "b", "a"],
            {"ab": 1},
            hard_cuts=[3],
            expected_error="InvalidConfiguration",
            seed=seed,
        ),
        _case(
            "error_classification",
            ["a", "b"],
            {"ab": 1},
            bad_atom=True,
            expected_error="InvalidConfiguration",
            seed=seed,
        ),
        _case(
            "error_classification",
            ["a", "b", "a", "b"],
            {"ab": 1},
            tape=[False],
            expected_error="DecisionError",
            seed=seed,
        ),
    ]


def _gen_deterministic_fixtures() -> List[Case]:
    return [
        _case("deterministic_fixtures", [], {}),
        _case("deterministic_fixtures", ["a"], {"a": 1}),
        _case("deterministic_fixtures", ["a", "b", "a", "b", "a", "b"], {"ab": 1}),
        # Leftmost overlap resolution: "aaa" with {aa} must pair leaves (0,1).
        _case("deterministic_fixtures", ["a", "a", "a"], {"aa": 1}),
        # Pass barrier: freshly merged groups cannot merge again in-pass.
        _case("deterministic_fixtures", ["a", "a", "a", "a"], {"aa": 1, "aaaa": 2}),
        # Hard cut splitting the input in half.
        _case("deterministic_fixtures", ["a", "a", "a", "a"], {"aa": 1}, hard_cuts=[2]),
        # Special tokens and byte-fallback-looking pieces are plain keys.
        _case(
            "deterministic_fixtures",
            ["<|endoftext|>", "<|endoftext|>", "a", "\u00ff"],
            {"<|endoftext|><|endoftext|>": 9},
        ),
        # Empty pieces are ordinary leaves; membership alone decides.
        _case("deterministic_fixtures", ["", "a", ""], {"a": 1, "": 2}),
        _case("deterministic_fixtures", ["\u4e16", "\u754c", "\u4e16\u754c"], {"\u4e16\u754c": 3}),
    ]


def _all_categories(master_rng: random.Random) -> List[Tuple[str, List[Case]]]:
    seeded = {
        "random_sequences": _gen_random_sequences,
        "conflicting_ranks": _gen_conflicting_ranks,
        "boundary_constraints": _gen_boundary_constraints,
        "byte_special_tokens": _gen_byte_special,
        "empty_and_repeated": _gen_empty_repeated,
        "unicode_sequences": _gen_unicode,
        "dropout_invalidation": _gen_dropout_invalidation,
    }
    categories: List[Tuple[str, List[Case]]] = [
        ("deterministic_fixtures", _gen_deterministic_fixtures()),
        ("error_classification", _gen_error_classification(MASTER_SEED)),
        ("pathological_long", _gen_pathological_long(MASTER_SEED)),
    ]
    for name, generator in seeded.items():
        per_case_seeds = [master_rng.randrange(2**32) for _ in range(CASES_PER_RANDOM_CATEGORY)]
        cases: List[Case] = []
        for case_seed in per_case_seeds:
            cases.extend(generator(case_seed))
        categories.append((name, cases))
    return categories


# ---------------------------------------------------------------------------
# Machine-readable artifact (issue #110 acceptance criterion)
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _write_artifact(categories: List[Tuple[str, Dict[str, Any]]], all_cases: List[Case]) -> Path:
    report = {
        "schema": "uniqtoken.merge_engine_differential/1.0",
        "issue": "110",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": _git_commit(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "configuration": {
            "master_seed": MASTER_SEED,
            "cases_per_random_category": CASES_PER_RANDOM_CATEGORY,
            "semantic_profile": SemanticProfile.SUPER_BPE_PASS_V1.value,
            "engines": ["reference", "fast"],
            "pathological_length": PATHOLOGICAL_LENGTH,
            # Both engines on this path are pure Python; there are no
            # release/debug build variants to cover. The native prototype
            # (issue #109) is out of scope for this suite.
            "build_coverage": "pure-python (no build variants on this path)",
        },
        "notes": "Correctness only: no performance conclusion may be drawn from this artifact.",
        "cases_sha256": hashlib.sha256(
            json.dumps(all_cases, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "categories": {name: stats for name, stats in categories},
        "result": {
            "total_cases": sum(stats["cases"] for _, stats in categories),
            "total_mismatches": sum(len(stats["mismatches"]) for _, stats in categories),
        },
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return ARTIFACT_PATH


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class MergeEngineDifferentialSweepTests(unittest.TestCase):
    """Run the full #110 differential sweep and emit the artifact."""

    def test_differential_sweep_zero_mismatches(self):
        master_rng = random.Random(MASTER_SEED)
        categories = _all_categories(master_rng)

        stats: List[Tuple[str, Dict[str, Any]]] = []
        all_cases: List[Case] = []
        failures: List[str] = []
        for name, cases in categories:
            mismatches: List[Dict[str, Any]] = []
            seeds = sorted({case["seed"] for case in cases if case["seed"] is not None})
            for case in cases:
                all_cases.append(case)
                match, reason = _compare_case(case)
                if not match:
                    shrunk = _shrink_pieces(case)
                    mismatches.append({"case": case, "shrunk_pieces": shrunk, "reason": reason})
                    failures.append(
                        f"[{name}] seed={case['seed']} reason={reason}\n"
                        f"minimal reproducible case:\n{_format_repro(case, shrunk)}"
                    )
            stats.append(
                (
                    name,
                    {"cases": len(cases), "mismatches": mismatches, "seeds": seeds},
                )
            )

        artifact = _write_artifact(stats, all_cases)
        total = sum(s["cases"] for _, s in stats)
        self.assertEqual(
            failures,
            [],
            msg=(
                f"{len(failures)} differential mismatches over {total} cases "
                f"(artifact: {artifact}); minimal reproducible cases follow:\n" + "\n---\n".join(failures[:5])
            ),
        )
        # The artifact must exist, be loadable, and record full case coverage.
        loaded = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertEqual(loaded["result"]["total_cases"], total)
        self.assertEqual(loaded["result"]["total_mismatches"], 0)
        self.assertNotEqual(loaded["commit"], "")
        self.assertIn("categories", loaded)


class MergeEngineDifferentialErrorContractTests(unittest.TestCase):
    """Error/unsupported-case classification equality (issue #110)."""

    def test_error_cases_classify_identically(self):
        for case in _gen_error_classification(MASTER_SEED):
            with self.subTest(category=case["category"], seed=case["seed"]):
                match, reason = _compare_case(case)
                self.assertTrue(match, msg=reason)

    def test_decision_tape_exhaustion_consumption_matches(self):
        # Same input, tape of exactly one draw against a 2-merge input:
        # both engines must raise DecisionError after consuming exactly 1 draw.
        case = _case(
            "error_classification",
            ["a", "b", "a", "b"],
            {"ab": 1},
            tape=[False],
            expected_error="DecisionError",
            seed=MASTER_SEED,
        )
        table, constraints, atoms = _build_fixtures(case)
        for engine in (ReferenceMergeEngine(), FastMergeEngine()):
            decisions = BooleanTapeDecisions(list(case["tape"]))
            with self.assertRaises(DecisionError):
                engine.apply(atoms, table, constraints, decisions)
            self.assertEqual(decisions.consumed, 1)


class FailureReductionTests(unittest.TestCase):
    """The shrinker must emit a minimal reproducible case (issue #110)."""

    def test_shrinker_reduces_to_minimal_failing_prefix(self):
        # Simulate a mismatch that only occurs while 3+ pieces remain, using
        # a predicate in place of the real comparison. The shrinker must end
        # at exactly 3 pieces.
        pieces = ["a"] * 20

        def failing(candidate: List[str]) -> bool:
            return len(candidate) >= 3

        # Re-implement the shrink loop against the synthetic predicate by
        # driving _shrink_pieces with a patched comparison.
        from unittest.mock import patch

        def fake_compare(case: Case) -> Tuple[bool, str]:
            return (not failing(case["pieces"]), "")

        with patch(f"{__name__}._compare_case", side_effect=fake_compare):
            shrunk = _shrink_pieces({"pieces": pieces})
        self.assertEqual(len(shrunk), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
