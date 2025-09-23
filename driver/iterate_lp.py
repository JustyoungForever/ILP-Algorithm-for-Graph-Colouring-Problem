from typing import Dict, Any, List, Set, Tuple
import time

from graph.clique import greedy_max_clique
from graph.verify import verify_coloring
from graph.local_search import consolidate_colors
from graph.cliques_enum import top_maximal_cliques
from graph.ub1_greedy import try_ub_minus_one_greedy

from ilp.model import build_lp_model
from ilp.lp_solve import solve_lp_and_extract
from ilp.fixing import choose_colors_after_fixing
from heuristics.round_and_repair import round_and_repair_multi


def _bootstrap_initial_from_lp(
    G,
    headroom: int = 3,
    max_k_increase: int = None,
    restarts: int = 16,
    time_limit_bootstrap: float = 10.0,
    verbose: bool = True,
) -> Tuple[int, int, List[int], List[List[int]], Dict[int, int], float]:
    """
    Use LP relaxation + rounding/repair to obtain the FIRST feasible coloring as the initial UB.
    - Start from LB = |clique| and set K = LB + headroom;
    - If rounding fails, increase K and retry (until success or limits).
    Returns: (UB, LB, clique_nodes, extra_cliques, best_coloring, z_LP_at_success)
    """
    t0 = time.time()
    n = G.number_of_nodes()

    clique_nodes = greedy_max_clique(G)
    LB = len(clique_nodes)

    # clique cuts to strengthen the LP
    extra_cliques = top_maximal_cliques(G, max_cliques=50, min_size=max(4, LB + 1), time_limit_sec=2.0)

    # start with K = LB + headroom and increase if rounding fails
    K = min(n, max(LB, LB + headroom))
    if max_k_increase is None:
        max_k_increase = max(0, n - K)

    last_zLP = None
    for bump in range(max_k_increase + 1):
        if time.time() - t0 > time_limit_bootstrap:
            break

        allowed_colors = list(range(K))
        # Build and solve the LP
        model, var_maps = build_lp_model(
            G=G,
            allowed_colors=allowed_colors,
            clique_nodes=clique_nodes,
            extra_cliques=extra_cliques,
            add_precedence=True,
        )
        info = solve_lp_and_extract(model, var_maps)
        last_zLP = info["z_LP"]

        # multi-start rounding + repair
        cand = round_and_repair_multi(G, info["x_frac"], info["y_frac"], current_UB=K, restarts=restarts, seed=bump)
        rep = verify_coloring(G, cand, allowed_colors=allowed_colors)

        if verbose:
            print(f"[Init-LP] try K={K} -> feasible={rep['feasible']} | used={len(set(cand.values()))} | z_LP={last_zLP:.6f}")
            if not rep["feasible"]:
                print(f"[Init-LP] conflicts_sample={rep['conflicts_sample'][:10]}")

        if rep["feasible"]:
            UB0 = len(set(cand.values()))
            return UB0, LB, clique_nodes, extra_cliques, cand, last_zLP

        # otherwise increase K and try once again
        K = min(n, K + 1)

    # fallback: try K = n one last time (should almost always succeed)
    if K < n:
        allowed_colors = list(range(n))
        model, var_maps = build_lp_model(
            G=G, allowed_colors=allowed_colors, clique_nodes=clique_nodes,
            extra_cliques=extra_cliques, add_precedence=True
        )
        info = solve_lp_and_extract(model, var_maps)
        last_zLP = info["z_LP"]
        cand = round_and_repair_multi(G, info["x_frac"], info["y_frac"], current_UB=n, restarts=restarts, seed=999)
        rep = verify_coloring(G, cand, allowed_colors=allowed_colors)
        if verbose:
            print(f"[Init-LP] fallback K=n -> feasible={rep['feasible']} | used={len(set(cand.values()))} | z_LP={last_zLP:.6f}")
        if rep["feasible"]:
            UB0 = len(set(cand.values()))
            return UB0, LB, clique_nodes, extra_cliques, cand, last_zLP

    #maybe not happen: return an empty coloring (will be caught by assertions later)
    return n, LB, clique_nodes, extra_cliques, {}, (last_zLP if last_zLP is not None else float("inf"))


def try_shrink_UB_by_one_LP(
    G, UB: int, clique_nodes: List[int], extra_cliques: List[List[int]], seed: int = 0, verbose: bool = False
):
    """LP with K=UB-1 + multi-start rounding; success -> return (UB-1, coloring, True)."""
    if UB <= len(clique_nodes):
        return UB, None, False
    K2 = list(range(UB - 1))
    model2, var_maps2 = build_lp_model(
        G, allowed_colors=K2, clique_nodes=clique_nodes, extra_cliques=extra_cliques, add_precedence=True
    )
    info2 = solve_lp_and_extract(model2, var_maps2)
    cand = round_and_repair_multi(G, info2["x_frac"], info2["y_frac"], current_UB=UB - 1, restarts=8, seed=seed)
    rep = verify_coloring(G, cand, allowed_colors=K2)
    if rep["feasible"]:
        if verbose:
            print(f"  [UB-1 Test (LP)] success -> UB={UB-1}")
        return UB - 1, cand, True
    else:
        if verbose:
            print(f"  [UB-1 Test (LP)] failed")
        return UB, None, False


def run_iterative_lp(
    G,
    time_limit_sec: int = 60,
    max_rounds: int = 200,
    stall_rounds: int = 10,
    min_rounds: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    iterative LP-based heuristic (first feasible solution from LP rounding):
    - bootstrap: start from LB; build LP with K=LB+headroom and use rounding/repair for initial feasible UB0.
    - iteration: each round runs LP - rounding -> conservative fixing (K=0..UB-1) - local search - UB-1 (graph) - UB-1 (LP).
    - stop when UB==LB, or stalled, or time/round limits are hit.
    """
    t_all0 = time.time()

    #Bootstrap: use LP rounding to get the first feasible coloring
    UB, LB, clique_nodes, extra_cliques, best_coloring, zLP0 = _bootstrap_initial_from_lp(
        G, headroom=3, max_k_increase=None, restarts=16, time_limit_bootstrap=min(10.0, time_limit_sec * 0.2), verbose=verbose
    )
    allowed_colors: List[int] = list(range(UB))  # Keep K aligned with UB 
    reserved_colors: Set[int] = set(range(LB))

    if verbose:
        print(f"[Init] (from LP-rounding) UB0 = {UB}, LB = {LB}, clique = {clique_nodes}, z_LP0 = {zLP0:.6f}")

    # maain loop
    logs: List[Dict[str, Any]] = []
    no_change_rounds = 0
    stop_reason = ""

    for it in range(1, max_rounds + 1):
        if time.time() - t_all0 > time_limit_sec:
            stop_reason = "time_limit"
            break

        improved = False

        # step 1: build and solve LP (with precedence + clique cuts)
        model, var_maps = build_lp_model(
            G=G, allowed_colors=allowed_colors, clique_nodes=clique_nodes,
            extra_cliques=extra_cliques, add_precedence=True
        )
        info = solve_lp_and_extract(model, var_maps)
        z_LP, x_frac, y_frac = info["z_LP"], info["x_frac"], info["y_frac"]
        rc_y = info["rc_y"]
        if verbose:
            print(f"[Iter {it}] z_LP={z_LP:.6f} |K|={len(allowed_colors)}")

        #step 2: multi-start rounding -> verify
        rounding_coloring = round_and_repair_multi(G, x_frac, y_frac, current_UB=len(allowed_colors), restarts=16, seed=it)
        rep = verify_coloring(G, rounding_coloring, allowed_colors=list(range(len(allowed_colors))))
        if verbose:
            print(f"  [Iter {it} / Rounding] feasible={rep['feasible']}|used_colors={len(set(rounding_coloring.values()))}|conflicts={rep['num_conflicts']}")
            if not rep["feasible"]:
                print(f"  [Iter {it} / Rounding] conflicts_sample ={rep['conflicts_sample'][:10]}")

        if rep["feasible"]:
            best_coloring = rounding_coloring
            used_colors_round = len(set(best_coloring.values()))
            if used_colors_round < UB:
                UB = used_colors_round
                allowed_colors = list(range(UB))
                improved = True
                if verbose:
                    print(f"  [Round+Repair] Improved UB -> {UB}")
        # if not feasible, keep previous best_coloring as a safety net.

        #step 3: conservative fixing (keep K = 0..UB-1)
        new_allowed = choose_colors_after_fixing(
            allowed_colors=allowed_colors, rc_y=rc_y, z_LP=z_LP, UB=UB, reserved_colors=reserved_colors
        )
        if set(new_allowed) != set(allowed_colors):
            allowed_colors = new_allowed
            improved = True
            if verbose:
                print(f"  [Fixing] K changed to {allowed_colors}")

        #local search: consolidate highest color classes
        best_coloring, reduced_flag = consolidate_colors(G, best_coloring, passes=5)
        if reduced_flag:
            newUB = len(set(best_coloring.values()))
            if newUB < UB:
                UB = newUB
                allowed_colors = list(range(UB))
                improved = True
                if verbose:
                    print(f"  [LocalSearch] Reduced to UB={UB}")

        #pure graph-domain UB-1 attempt
        if not improved:
            cand, ok = try_ub_minus_one_greedy(G, best_coloring)
            if ok:
                best_coloring = cand
                UB = len(set(best_coloring.values()))
                allowed_colors = list(range(UB))
                improved = True
                if verbose:
                    print(f"  [UB-1 Greedy] success -> UB={UB}")

        #LP(UB-1) attempt
        if not improved:
            newUB, newcol, ok = try_shrink_UB_by_one_LP(G, UB, clique_nodes, extra_cliques, seed=it, verbose=verbose)
            if ok:
                UB = newUB
                allowed_colors = list(range(UB))
                best_coloring = newcol
                improved = True

        # Logs and invariants
        logs.append(dict(it=it, UB=UB, LB=LB, z_LP=z_LP, K=len(allowed_colors)))

        assert UB >= LB, "Invariant broken: UB < LB"
        assert set(allowed_colors) == set(range(len(allowed_colors))), "Invariant broken: K must be contiguous 0..K-1"
        assert z_LP <= UB + 1e-6, "LP lower bound exceeds UB"
        assert verify_coloring(G, best_coloring, allowed_colors=list(range(UB)))["feasible"], "Best coloring infeasible"

        #stopping criteria
        if UB <= LB:
            stop_reason = "UB==LB"
            if verbose:
                print("  [Stop] UB==LB")
            break

        no_change_rounds = 0 if improved else (no_change_rounds + 1)
        if no_change_rounds >= stall_rounds and it >= min_rounds:
            stop_reason = f"no_progress {no_change_rounds} rounds"
            if verbose:
                print(f"  [Stop] {stop_reason}")
            break

        if time.time() - t_all0 > time_limit_sec:
            stop_reason = "time_limit"
            break

    if not stop_reason:
        stop_reason = "max_rounds"

    final_report = verify_coloring(G, best_coloring, allowed_colors=list(range(UB)))
    if verbose:
        print(f"  [Final] feasible={final_report['feasible']}|used_colors={UB}|conflicts={final_report['num_conflicts']}")

    return dict(
        UB=UB, LB=LB, coloring=best_coloring, iters=len(logs), log=logs,
        stop_reason=stop_reason, feasible=final_report["feasible"], final_check=final_report
    )
