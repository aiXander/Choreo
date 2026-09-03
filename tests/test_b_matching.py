"""Pins for ``greedy_b_matching``'s force-fill (phase 3) — the behaviour that
decides a member's guaranteed b_min-th card.

The load-bearing invariant: by the time phase 3 runs for a member, EVERY
remaining candidate partner sits at or above its cap (phase 2 already took
any partner that was under it). So "lowest partner degree first, weight
second" means: best remaining pair among partners at exactly ``b_max`` (each
reaches at most ``b_max + 1``), and only when no such partner exists does the
fill escalate one degree level, spreading overflow rather than piling it onto
one popular profile. Both halves are pinned below; changing either sort key
silently changes who receives the weakest cards.
"""

from choreo.match import greedy_b_matching
from choreo.schemas import Edge
from choreo.utils import stable_pair_id


def _edge(a: str, b: str, w: float) -> Edge:
    u1, u2 = min(a, b), max(a, b)
    return Edge(user1=u1, user2=u2, pair_id=stable_pair_id(u1, u2),
                final_weight=w, embed_score=w, llm_score=w)


def _ids(edges):
    return {e.pair_id for e in edges}


def test_force_fill_takes_best_remaining_pair_at_the_soft_cap():
    # b_max=1: p1 and p2 both fill their one slot in phase 1, leaving m with
    # nothing; phase 3 must give m its best partner (p1, 0.8), not the
    # lighter-weight p2, and p1 ends at exactly b_max + 1.
    edges = [_edge("p1", "x", 0.9), _edge("p2", "y", 0.5),
             _edge("m", "p1", 0.8), _edge("m", "p2", 0.3)]
    users = {"m", "p1", "p2", "x", "y"}
    selected = greedy_b_matching(edges, b_min=1, b_max=1, all_users=users)
    ids = _ids(selected)
    assert stable_pair_id("m", "p1") in ids
    assert stable_pair_id("m", "p2") not in ids
    degree = {u: sum(1 for e in selected if u in (e.user1, e.user2)) for u in users}
    assert degree["p1"] == 2  # b_max + 1, never more while alternatives exist
    assert degree["m"] == 1   # b_min met


def test_force_fill_never_stacks_a_partner_past_b_max_plus_one_while_another_can_take_it():
    # m1's force-fill pushes p1 to b_max + 1. m2 then prefers p2 (still at
    # b_max) over the better-scoring but already-overflowed p1.
    edges = [_edge("p1", "x", 0.95), _edge("p2", "y", 0.5),
             _edge("m1", "p1", 0.9), _edge("m2", "p1", 0.85), _edge("m2", "p2", 0.3)]
    users = {"m1", "m2", "p1", "p2", "x", "y"}
    selected = greedy_b_matching(edges, b_min=1, b_max=1, all_users=users)
    ids = _ids(selected)
    assert stable_pair_id("m1", "p1") in ids
    assert stable_pair_id("m2", "p2") in ids
    assert stable_pair_id("m2", "p1") not in ids
    degree = {u: sum(1 for e in selected if u in (e.user1, e.user2)) for u in users}
    assert degree["p1"] == 2 and degree["p2"] == 2
    assert degree["m1"] == 1 and degree["m2"] == 1
