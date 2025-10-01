# driver/accelerators.py
from typing import Dict, Tuple
from graph.verify import verify_coloring
from graph.local_search import consolidate_colors, lp_guided_pack_highest_color
from heuristics.round_and_repair import round_and_repair_multi
from graph.ub1_greedy import try_ub_minus_one_greedy

def compact_colors(coloring: Dict[int, int]) -> Tuple[Dict[int, int], int]:
    """remap colors to 0..k-1, returns (new_coloring, number_of_colors_used)."""
    used = sorted(set(coloring.values()))
    remap = {c: i for i, c in enumerate(used)}
    if all(remap[c] == c for c in used):
        return dict(coloring), len(used)
    newc = {v: remap[c] for v, c in coloring.items()}
    return newc, len(used)

def try_side_rounding(
    G, x_frac, y_frac, K: int, UB: int, *,
    restarts: int, perturb_y: float, seed: int, verbose: bool = False
) -> Tuple[int, Dict[int,int], bool]:
    """
    temporarily use K+1 colors for rounding while keeping LP locked at K;
    returns improvement if a smaller UB is constructed.
    """
    side_colors = K + 1
    if verbose and side_colors > UB:
        print(f"  [Side] skip: K+1={side_colors} exceeds current UB={UB}")
    if side_colors > UB:
        return UB, {}, False
    cand = round_and_repair_multi(
        G, x_frac, y_frac, current_UB=side_colors,
        restarts=max(4, restarts // 2), seed=seed, perturb_y=perturb_y
    )
    rep = verify_coloring(G, cand, allowed_colors=list(range(side_colors)))
    if verbose and not rep["feasible"]:
        print("  [Side] K+1 rounding produced an infeasible coloring (will not update UB)")
        return UB, {}, False
    # Compact + small local search
    cand, used = compact_colors(cand)
    cand2, reduced = consolidate_colors(G, dict(cand), passes=3)
    if reduced:
        cand2, used = compact_colors(cand2)
        cand = cand2
    if used < UB:
        if verbose:
            print(f"  [UB-1 side attempt] improved UB -> {used} (using {side_colors} colors)")
        return used, cand, True
    if verbose and used >= UB:
        print(f"  [Side] feasible but not better (used={used} ≥ UB={UB})")
    return UB, {}, False

def try_lp_guided_pack(
    G, best_coloring: Dict[int,int], x_frac, K: int, UB: int, *, verbose: bool = False
) -> Tuple[int, Dict[int,int], bool]:
    """
    only called when UB == K+1: use LP's x_frac to guide packing of highest color layer (K) back into 0..K-1.
    success means UB becomes <= K.
    """
    if verbose:
        print(f"  [LP-guided pack] trying to empty highest color K={K} (only active if UB==K+1)")

    if UB != K + 1:
        return UB, {}, False
    packed, shr = lp_guided_pack_highest_color(G, best_coloring, x_frac, K, passes=2)
    if verbose and not shr:
        print("  [LP-guided pack] no full emptying of highest color — no UB reduction")
        return UB, {}, False
    packed, used = compact_colors(packed)
    if used < UB:
        if verbose:
            print(f"  [LP-guided pack] UB -> {used}")
        return used, packed, True
    return UB, {}, False

def try_graph_ub1_greedy(
    G, best_coloring: Dict[int,int], UB: int, *, verbose: bool = False
) -> Tuple[int, Dict[int,int], bool]:
    """
    pure graph-domain UB-1 greedy (complementary to LP-guided). Success reduces UB by 1.
    """
    if verbose:
        print("  [UB-1 greedy] attempting to fold top color class graphically")
    cand, ok = try_ub_minus_one_greedy(G, best_coloring)
    if not ok:
        return UB, {}, False
    cand, used = compact_colors(cand)
    if used < UB:
        if verbose:
            print(f"  [UB-1 greedy] UB -> {used}")
        return used, cand, True
    if verbose and used >= UB:
        print(f"  [UB-1 greedy] feasible but not better (used={used} >= UB={UB})")
    return UB, {}, False