# driver/iterate_lp.py
from typing import Dict, Any, List, Set, Tuple
import time
import math
from graph.clique import greedy_max_clique
from graph.verify import verify_coloring
from graph.local_search import consolidate_colors, lp_guided_pack_highest_color
from graph.cliques_enum import top_maximal_cliques
from graph.ub1_greedy import try_ub_minus_one_greedy
from graph.dsatur import dsatur_coloring
from ilp.model import build_lp_model
from ilp.lp_solve import solve_lp_and_extract
from ilp.fixing import choose_colors_after_fixing
from heuristics.round_and_repair import round_and_repair_multi
from driver.accelerators import (
    compact_colors,
    try_side_rounding,
    try_lp_guided_pack,
    try_graph_ub1_greedy,
)

from ilp.incremental import IncrementalLP
from ilp.fixing_policies import pick_fixings
from graph.slo import smallest_last_coloring

def run_iterative_lp_v2(
    G,
    time_limit_sec: int = 60,
    verbose: bool = True,
    init_heuristic: str = "dsatur",   # or "smallest_last"
    fix_policy: str = "prefix_shrink+strong_assign",
    strong_margin: float = 0.25,
    max_fix_per_round: int = 50,
    restarts: int = 16,
    perturb_y: float = 1e-6,
) -> Dict[str, Any]:
    t0 = time.time()
    stall_after_K = 8  # Trigger Step-5 termination when consecutive rounds without progress reach this value and K==ceil(zLP)
    
    # Schritt 0: LB + Initial solution (and compact color indices)
    clique_nodes = greedy_max_clique(G)
    LB = len(clique_nodes)
    init_col = smallest_last_coloring(G) if init_heuristic == "smallest_last" else dsatur_coloring(G)
    init_col, UB = compact_colors(init_col)   # Compact color indices to 0..UB-1
    best_coloring = init_col
    K = UB  # Initialize LP/rounding palette K, later tightened by ceil(zLP)
    if verbose:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        print(f"[Init] graph: |V|={n} |E|={m} |LB|={LB} |UB(init)|={UB} |K|={K} via {init_heuristic}")

    if verbose:
        print(f"[Init] LB={LB} | UB(init:{init_heuristic})={UB}")
    if UB <= LB:
        rep = verify_coloring(G, best_coloring, allowed_colors=list(range(UB)))
        return dict(UB=UB, LB=LB, coloring=best_coloring, iters=0, log=[], stop_reason="UB==LB",
                    feasible=rep["feasible"], final_check=rep)

    # build LP once (initial K = UB)
    extra_cliques = top_maximal_cliques(G, max_cliques=50, min_size=max(4, LB+1), time_limit_sec=2.0)
    inc = IncrementalLP(G, allowed_colors=list(range(UB)), clique_nodes=clique_nodes,
                        extra_cliques=extra_cliques, add_precedence=True)
    if verbose:
        print(f"[LP] model built once (incremental), initial |C|={len(inc.C)} (K={K}) with precedence & clique-cuts")

    K = UB  # palette size used for LP/rounding/verification; different from UB (best feasible coloring found)
    logs = []
    round_id = 0
    stop_reason = ""
    prev_UB = UB
    no_progress_rounds = 0

    while time.time() - t0 <= time_limit_sec:
        round_id += 1

        # 1) solve LP first (get zLP)
        info = inc.solve()
        zLP, x_frac, y_frac = info["z_LP"], info["x_frac"], info["y_frac"]
        if verbose:
            ceilK = int(math.ceil(zLP - 1e-12))
            elapsed = time.time() - t0
            print(f"[Round {round_id}] zLP={zLP:.6f} | ceil(zLP)={ceilK} | K={K} | UB={UB} | t={elapsed:.2f}s")
        # 2) decide whether to tighten K based on ceil(zLP) (safe: try - rollback if fails)
        K_target = max(LB, int(math.ceil(zLP - 1e-12)))
        K_shrunk_this_round = False
        x_fix_applied_this_round = False
        improved_this_round = False

        if K_target < K:
            tok = [inc.lock_prefix_K(K_target)]
            ok_info, ok = inc.try_apply_and_solve(tok)
            if ok:
                K = K_target
                K_shrunk_this_round = True
                if verbose:
                    print(f"  [Fix] shrink K to {K} (ceil(zLP)={K})")
                # continue with updated solution to avoid re-solving
                zLP, x_frac, y_frac = ok_info["z_LP"], ok_info["x_frac"], ok_info["y_frac"]
            else:
                inc.revert_all(tok)
                if verbose:
                    print(f"  [Fix] shrink to K={K_target} infeasible - rollback; keep K={K}")

            # 3) rounding with K colors
            cand = round_and_repair_multi(G, x_frac, y_frac, current_UB=K,
                                        restarts=restarts, seed=round_id, perturb_y=perturb_y)
            rep = verify_coloring(G, cand, allowed_colors=list(range(K)))

            if rep["feasible"]:
                # (A) feasible path: maybe improve UB, sync K, etc.
                cand, used = compact_colors(cand)
                cand2, reduced = consolidate_colors(G, dict(cand), passes=5)
                if reduced:
                    cand2, used = compact_colors(cand2)
                    cand = cand2
                if used < UB:
                    UB = used
                    best_coloring = cand
                    improved_this_round = True
                    if verbose:
                        print(f"  [Rounding/Local] improved UB -> {UB}")
                    if K > UB:
                        tok2 = [inc.lock_prefix_K(UB)]
                        ok_info2, ok2 = inc.try_apply_and_solve(tok2)
                        if ok2:
                            K = UB
                            if verbose:
                                print(f"  [Sync] K aligned to new UB: K={K}")
                            zLP, x_frac, y_frac = ok_info2["z_LP"], ok_info2["x_frac"], ok_info2["y_frac"]
                        else:
                            inc.revert_all(tok2)
                    if UB <= LB:
                        stop_reason = "UB==LB"
                        break
            else:
                # (B) infeasible path: try K+1 side rounding before any fixing
                if verbose:
                    print(f"  [Side] trying K+1 rounding (K={K} - {K+1}) to break stalemate")
                newUB, newCol, applied = try_side_rounding(
                    G, x_frac, y_frac, K, UB,
                    restarts=restarts, perturb_y=perturb_y,
                    seed=9000 + round_id, verbose=verbose
                )
                if applied and newUB < UB:
                    UB = newUB
                    best_coloring = newCol
                    improved_this_round = True
                    if UB <= LB:
                        stop_reason = "UB==LB"
                        break

            if verbose:
                used_try = len(set(cand.values()))
                print(f"  [Rounding] feasible={rep['feasible']} | used={used_try} | conflicts={rep['num_conflicts']}")

            # 3.5) UB-1 side attempt (allow K+1 colors, only for constructing better feasible solutions; doesn't change LP locking)
            newUB, newCol, applied = try_side_rounding(
                G, x_frac, y_frac, K, UB,
                restarts=restarts, perturb_y=perturb_y, seed=9000 + round_id, verbose=verbose
            )
            if applied and newUB < UB:
                UB = newUB
                best_coloring = newCol
                improved_this_round = True
                if UB <= LB:
                    stop_reason = "UB==LB"
                    break
        if verbose:
            used_try = len(set(cand.values()))
            print(f"  [Rounding] feasible={rep['feasible']} | used={used_try} | conflicts={rep['num_conflicts']}")
        #  and LP-guided pack: when UB == K+1, pack highest color layer back into 0..K-1
        if verbose and UB == K + 1:
            print(f"  [PackWindow] UB==K+1 (UB={UB}, K={K}) - attempt LP-guided pack and UB-1 greedy")

        if UB == K + 1:
            # B1) LP-guided highest color layer packing
            newUB, newCol, applied = try_lp_guided_pack(G, best_coloring, x_frac, K, UB, verbose=verbose)
            if applied and newUB < UB:
                UB = newUB
                best_coloring = newCol
                improved_this_round = True
                if K > UB:
                    tok3 = [inc.lock_prefix_K(UB)]
                    ok_info3, ok3 = inc.try_apply_and_solve(tok3)
                    if ok3:
                        K = UB
                        if verbose:
                            print(f"  [Sync] K aligned to new UB: K={K}")
                    else:
                        inc.revert_all(tok3)
                if UB <= LB:
                    stop_reason = "UB==LB"
                    break
            # B2) pure graph-domain UB-1 greedy 
            newUB, newCol, applied = try_graph_ub1_greedy(G, best_coloring, UB, verbose=verbose)
            if applied and newUB < UB:
                UB = newUB
                best_coloring = newCol
                improved_this_round = True
                if K > UB:
                    tok4 = [inc.lock_prefix_K(UB)]
                    ok_info4, ok4 = inc.try_apply_and_solve(tok4)
                    if ok4:
                        K = UB
                        if verbose:
                            print(f"  [Sync] K aligned to new UB: K={K}")
                    else:
                        inc.revert_all(tok4)
                if UB <= LB:
                    stop_reason = "UB==LB"
                    break

        # 4) maybe other fixing policies (strong x fixing, etc.), also follow safe process (only in prefix color domain)
        plan = pick_fixings(
            G=G, x_frac=x_frac, y_frac=y_frac, z_LP=zLP, UB=K, LB=LB,
            policy=fix_policy, strong_margin=strong_margin,
            max_fix_per_round=max_fix_per_round,
            rounding_coloring=cand if rep["feasible"] else None
        )
        if verbose:
            print(f"  [FixPolicy] y_zero={len(plan['y_zero'])} | x_one={len(plan['x_one'])} | x_zero={len(plan['x_zero'])} "
                f"| strong_margin={strong_margin} | max_fix_per_round={max_fix_per_round}")

        tokens = []
        for (v, c) in plan["x_one"]:
            if c < K:
                tokens.append(inc.fix_vertex_color(v, c))

        if tokens:
            if verbose:
                print(f"  [FixApply] applying tokens={len(tokens)} (x fixings only in prefix K)")

            ok_info3, ok3 = inc.try_apply_and_solve(tokens)
            if verbose:
                print(f"  [FixApply] status={'OK' if ok3 else 'ROLLBACK'}")
            if not ok3:
                inc.revert_all(tokens)
            else:
                x_fix_applied_this_round = True
                cand3 = round_and_repair_multi(G, ok_info3["x_frac"], ok_info3["y_frac"],
                                               current_UB=K, restarts=max(2, restarts // 2),
                                               seed=7777 + round_id, perturb_y=perturb_y)
                rep3 = verify_coloring(G, cand3, allowed_colors=list(range(K)))
                if verbose:
                    used3_try = len(set(cand3.values()))
                    print(f"  [After-Fix Rounding] feasible={rep3['feasible']} | used={used3_try} | conflicts={rep3['num_conflicts']}")

                if rep3["feasible"]:
                    cand3, used3 = compact_colors(cand3)
                    cand3b, reduced3 = consolidate_colors(G, dict(cand3), passes=3)
                    if reduced3:
                        cand3b, used3 = compact_colors(cand3b)
                        cand3 = cand3b
                    if used3 < UB:
                        UB = used3
                        best_coloring = cand3


                        improved_this_round = True
                        if verbose:
                            print(f"  [After-Fix] improved UB -> {UB}")
                        if K > UB:
                            tok4 = [inc.lock_prefix_K(UB)]
                            ok_info4, ok4 = inc.try_apply_and_solve(tok4)
                            if ok4:
                                K = UB
                                if verbose:
                                    print(f"  [Sync] K aligned to new UB: K={K}")
                            else:
                                inc.revert_all(tok4)
                        if UB <= LB:
                            stop_reason = "UB==LB"
                            break
        if verbose:
            print(f"  [Round {round_id} END] UB={UB} | K={K} | ceil(zLP)={int(math.ceil(zLP - 1e-12))} "
                f"| improved={improved_this_round} | fixed={(K_shrunk_this_round or x_fix_applied_this_round)}")

        logs.append(dict(round=round_id, zLP=zLP, UB=UB, K=K))

        # 5) termination logic
        # (a) Theoretical limit: ceil(zLP) >= UB and K <= UB
        if int(math.ceil(zLP - 1e-12)) >= UB and K <= UB:
            if verbose:
                print("  [Stop] LP bound meets UB (ceil(zLP) >= UB and K <= UB)")
            stop_reason = "no_better_than_UB"
            break
        # (b) Practical stagnation: K already aligned with ceil(zLP), and several rounds without substantial progress or fixing
        made_fix = (K_shrunk_this_round or x_fix_applied_this_round)
        if (not improved_this_round) and (not made_fix):
            no_progress_rounds += 1
        else:
            no_progress_rounds = 0
        if (K == int(math.ceil(zLP - 1e-12))) and (no_progress_rounds >= stall_after_K):
            stop_reason = "stalled_at_K_eq_ceil_zLP"
            if verbose:
                print(f"  [Stop] stalled at K==ceil(zLP) for {no_progress_rounds} rounds")

            break


    if not stop_reason:
        stop_reason = "time_limit" if time.time() - t0 > time_limit_sec else "max_rounds"
    
    # Final compaction before returning, ensure UB matches color indices (prevent out_of_range)
    best_coloring, used_final = compact_colors(best_coloring)
    if used_final != UB:
        UB = used_final
        if K > UB:
            # noot strictly necessary, but for consistency (ok even without rollback since  just before return)
            pass
    final_report = verify_coloring(G, best_coloring, allowed_colors=list(range(UB)))
    if verbose:
        print(f"[Final] UB={UB} LB={LB} feasible={final_report['feasible']} zLP≈{zLP:.6f}")
    return dict(
        UB=UB, LB=LB, coloring=best_coloring, iters=len(logs), log=logs,
        stop_reason=stop_reason, feasible=final_report["feasible"], final_check=final_report
    )


def _bootstrap_initial_from_lp(
    G,
    headroom: int = 3,
    max_k_increase: int = None,
    restarts: int = 16,
    time_limit_bootstrap: float = 10.0,
    verbose: bool = True,
) -> Tuple[int, int, List[int], List[List[int]], Dict[int, int], float]:
    """
    use LP relaxation + rounding/repair to obtain the FIRST feasible coloring as the initial UB.
    - start from LB = |clique| and set K = LB + headroom;
    - iff rounding fails, increase K and retry (until success or limits).
    returns: (UB, LB, clique_nodes, extra_cliques, best_coloring, z_LP_at_success)
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
        rep = verify_coloring(G, cand, allowed_colors=list(range(K)))

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