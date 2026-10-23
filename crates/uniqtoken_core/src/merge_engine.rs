//! Private SuperBpePassV1 prototype. Repository-local contract, no FastBPE code.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

type Group = (usize, usize, String, Option<u64>);
// Pass, original left leaf, right node, left generation, right generation.
type Candidate = (usize, usize, usize, usize, usize);

#[derive(Debug)]
struct Node {
    end: usize,
    piece: String,
    merged_id: Option<u64>,
    prev: Option<usize>,
    next: Option<usize>,
    generation: usize,
    alive: bool,
}

#[derive(Debug)]
struct Plan {
    groups: Vec<Group>,
    applied_merges: usize,
    decisions_consumed: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct Failure {
    kind: &'static str,
    decisions_consumed: usize,
}

fn enqueue(queue: &mut BinaryHeap<Reverse<Candidate>>, nodes: &[Node], left: usize, pass: usize) {
    if let Some(right) = nodes[left].next {
        queue.push(Reverse((
            pass,
            left,
            right,
            nodes[left].generation,
            nodes[right].generation,
        )));
    }
}

fn partition_valid(groups: &[Group], leaves: usize, applied: usize) -> bool {
    let mut cursor = 0;
    for (start, end, _, id) in groups {
        if *start != cursor || end <= start || *end > leaves || (*end - *start > 1 && id.is_none())
        {
            return false;
        }
        cursor = *end;
    }
    cursor == leaves && leaves.checked_sub(groups.len()) == Some(applied)
}

fn apply(
    pieces: &[String],
    eligible: &HashMap<String, u64>,
    hard_cuts: &[usize],
    tape: Option<&[bool]>,
) -> Result<Plan, Failure> {
    let failure = |kind, decisions_consumed| Failure {
        kind,
        decisions_consumed,
    };
    let n = pieces.len();
    if hard_cuts.iter().any(|&cut| cut == 0 || cut >= n) {
        return Err(failure("InvalidConfiguration", 0));
    }
    let cuts: HashSet<_> = hard_cuts.iter().copied().collect();
    let mut nodes: Vec<_> = pieces
        .iter()
        .enumerate()
        .map(|(i, piece)| Node {
            end: i + 1,
            piece: piece.clone(),
            merged_id: None,
            prev: i.checked_sub(1),
            next: if i + 1 < n { Some(i + 1) } else { None },
            generation: 0,
            alive: true,
        })
        .collect();
    let mut queue = BinaryHeap::new();
    for left in 0..n.saturating_sub(1) {
        enqueue(&mut queue, &nodes, left, 0);
    }
    // One remembered identity per left slot: unchanged dropped boundaries do
    // not retry. Changed constituents have a different generation/neighbor.
    let mut visited = vec![None; n];
    let mut current_pass = 0;
    let mut applied = 0;
    let mut consumed = 0;
    while let Some(Reverse((pass, left, right, lg, rg))) = queue.pop() {
        if pass < current_pass
            || !nodes[left].alive
            || !nodes[right].alive
            || nodes[left].generation != lg
            || nodes[right].generation != rg
            || nodes[left].next != Some(right)
            || nodes[right].prev != Some(left)
        {
            continue;
        }
        let identity = (lg, right, rg);
        if visited[left] == Some(identity) {
            continue;
        }
        visited[left] = Some(identity);
        current_pass = pass;
        if cuts.contains(&nodes[left].end) {
            continue;
        }
        let mut result = nodes[left].piece.clone();
        result.push_str(&nodes[right].piece);
        let Some(&result_id) = eligible.get(&result) else {
            continue;
        };
        if let Some(decisions) = tape {
            let Some(&drop) = decisions.get(consumed) else {
                return Err(failure("DecisionError", consumed));
            };
            consumed += 1;
            if drop {
                continue;
            }
        }
        let Some(generation) = lg.checked_add(1) else {
            return Err(failure("InvariantViolation", consumed));
        };
        let Some(next_pass) = pass.checked_add(1) else {
            return Err(failure("InvariantViolation", consumed));
        };
        let next = nodes[right].next;
        nodes[left].piece = result;
        nodes[left].end = nodes[right].end;
        nodes[left].merged_id = Some(result_id);
        nodes[left].generation = generation;
        nodes[left].next = next;
        nodes[right].alive = false;
        nodes[right].piece = String::new();
        if let Some(next) = next {
            nodes[next].prev = Some(left);
        }
        applied += 1; // Bounded by n - 1: each merge retires one live node.
        if let Some(prev) = nodes[left].prev {
            enqueue(&mut queue, &nodes, prev, next_pass);
        }
        enqueue(&mut queue, &nodes, left, next_pass);
    }
    let mut groups = Vec::with_capacity(n - applied);
    let mut cursor = if n == 0 { None } else { Some(0) };
    while let Some(i) = cursor {
        if !nodes[i].alive || groups.len() >= n {
            return Err(failure("InvariantViolation", consumed));
        }
        groups.push((
            i,
            nodes[i].end,
            std::mem::take(&mut nodes[i].piece),
            nodes[i].merged_id,
        ));
        cursor = nodes[i].next;
    }
    if !partition_valid(&groups, n, applied) {
        return Err(failure("InvariantViolation", consumed));
    }
    Ok(Plan {
        groups,
        applied_merges: applied,
        decisions_consumed: consumed,
    })
}

#[cfg(feature = "python")]
#[pyo3::pyfunction]
#[pyo3(signature = (pieces, eligible, hard_cuts, tape=None))]
pub(crate) fn _experimental_merge_plan(
    py: pyo3::Python<'_>,
    pieces: Vec<String>,
    eligible: HashMap<String, u64>,
    hard_cuts: Vec<usize>,
    tape: Option<Vec<bool>>,
) -> (Vec<Group>, usize, usize, Option<&'static str>) {
    py.detach(
        move || match apply(&pieces, &eligible, &hard_cuts, tape.as_deref()) {
            Ok(plan) => (
                plan.groups,
                plan.applied_merges,
                plan.decisions_consumed,
                None,
            ),
            Err(error) => (Vec::new(), 0, error.decisions_consumed, Some(error.kind)),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        pieces: &[&str],
        rules: &[(&str, u64)],
        cuts: &[usize],
        tape: Option<&[bool]>,
    ) -> Result<Plan, Failure> {
        apply(
            &pieces.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            &rules.iter().map(|(s, id)| (s.to_string(), *id)).collect(),
            cuts,
            tape,
        )
    }

    #[test]
    fn leftmost_and_pass_barrier() {
        let plan = run(
            &["a", "b", "c", "d"],
            &[("ab", 9), ("abc", 1), ("cd", 2)],
            &[],
            None,
        )
        .unwrap();
        assert_eq!(
            plan.groups,
            vec![(0, 2, "ab".into(), Some(9)), (2, 4, "cd".into(), Some(2))]
        );
        assert_eq!(plan.applied_merges, 2);
        assert_eq!(plan.decisions_consumed, 0);
        let plan = run(&["a", "b", "c"], &[("ab", 9), ("bc", 1)], &[], None).unwrap();
        assert_eq!(plan.groups[0], (0, 2, "ab".into(), Some(9)));
    }

    #[test]
    fn changed_neighbor_releases_dropout_and_stale_entries_do_not_draw() {
        let plan = run(
            &["a", "b", "c"],
            &[("ab", 1), ("bc", 2), ("abc", 3)],
            &[],
            Some(&[true, false, false]),
        )
        .unwrap();
        assert_eq!(plan.groups, vec![(0, 3, "abc".into(), Some(3))]);
        assert_eq!(plan.decisions_consumed, 3);
        let plan = run(
            &["a", "b", "c"],
            &[("ab", 1), ("bc", 2)],
            &[],
            Some(&[false]),
        )
        .unwrap();
        assert_eq!(plan.decisions_consumed, 1);
    }

    #[test]
    fn unchanged_dropped_pair_never_retries_after_unrelated_merge() {
        let plan = run(
            &["a", "b", "c", "d", "e"],
            &[("ab", 1), ("de", 2)],
            &[],
            Some(&[true, false]),
        )
        .unwrap();
        assert_eq!(plan.decisions_consumed, 2);
        assert_eq!(plan.groups.len(), 4);
    }

    #[test]
    fn empty_pieces_terminate_and_cuts_remain_leaf_boundaries() {
        let plan = run(&["", "", "a", ""], &[("", 0), ("a", u64::MAX)], &[2], None).unwrap();
        assert_eq!(
            plan.groups,
            vec![
                (0, 2, "".into(), Some(0)),
                (2, 4, "a".into(), Some(u64::MAX))
            ]
        );
        assert!(run(&[], &[], &[], None).unwrap().groups.is_empty());
        assert_eq!(
            run(&[""], &[("", 7)], &[], None).unwrap().groups,
            vec![(0, 1, "".into(), None)]
        );
    }

    #[test]
    fn invalid_cuts_fail_before_decisions_and_exhaustion_reports_consumption() {
        let error = run(&["a", "b"], &[("ab", 1)], &[0], Some(&[])).unwrap_err();
        assert_eq!(
            error,
            Failure {
                kind: "InvalidConfiguration",
                decisions_consumed: 0
            }
        );
        let error = run(
            &["a", "b", "c"],
            &[("ab", 1), ("abc", 2)],
            &[],
            Some(&[false]),
        )
        .unwrap_err();
        assert_eq!(
            error,
            Failure {
                kind: "DecisionError",
                decisions_consumed: 1
            }
        );
    }

    #[test]
    fn malformed_partition_is_not_a_successful_plan() {
        assert!(!partition_valid(
            &[(0, 1, "a".into(), None), (2, 3, "b".into(), None)],
            3,
            1
        ));
        assert!(!partition_valid(&[(0, 2, "ab".into(), Some(1))], 2, 0));
        assert!(!partition_valid(&[(0, 0, "".into(), None)], 0, 0));
    }
}
