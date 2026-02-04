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
    _debug_candidate_from_xfrac,
)

from ilp.incremental import IncrementalLP
from ilp.fixing_policies import pick_fixings
from graph.slo import smallest_last_coloring
from visualisierung.draw import visualize_coloring

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
    enable_visualization: bool = False,
    viz_out_dir: str = "visualisierung/picture",
    viz_layout_seed: int = 42,
    algo_seed:int=0,
    edge_mode: str = "auto",
    lazy_threshold: float = 2e7,
) -> Dict[str, Any]:
    t0 = time.monotonic()
    deadline = t0 + float(time_limit_sec)
    print(f"[DBG-TL-init] time_limit_sec={time_limit_sec} deadline_in={deadline - t0:.1f}s")

    stop_reason = ""
    best_time_sec = 0.0
    best_round = 0
    best_UB_seen = None  
    zLP = float("nan")
    def elapsed():
        return time.monotonic() - t0

    def time_left():
        return deadline - time.monotonic()

    def budget_exhausted(stage: str) -> bool:
        nonlocal stop_reason
        if time_left() <= 0.0:
            stop_reason = "time_limit"
            if verbose:
                print(f"[Time] budget exhausted at {stage}, UB={UB if UB is not None else 'N/A'}")
            return True
        return False
    
    def mark_best(new_ub: int, rid: int) -> None:
        nonlocal best_UB_seen, best_time_sec, best_round
        if best_UB_seen is None or new_ub < best_UB_seen:
            best_UB_seen = new_ub
            best_time_sec = elapsed()
            best_round = rid
    def viz(step: str, rid: int, cmap: Dict[int, int], K_allowed: int, *, only_colored: bool = True) -> None:
        if not enable_visualization:
            return
        if not cmap:
            return
        visualize_coloring(
            G, cmap, step=step, round_id=rid,
            out_dir=viz_out_dir, layout_seed=viz_layout_seed,
            only_colored=only_colored,  # ← 动态控制
            allowed_colors=list(range(max(1, K_allowed))),
            conflict_nodes_fill_black=True,
        )

    def _accept_if_feasible(label: str, cand_map: Dict[int, int], ub_new: int) -> Tuple[bool, Dict[int, int], int, Dict[str, Any]]:
        """
        Returns (ok, cand_compact, used, report)
        Ensures cand is compacted and verified with allowed colors 0..used-1.
        """
        if not cand_map:
            return False, {}, ub_new, {"feasible": False, "num_conflicts": -1}

        cand_compact, used = compact_colors(cand_map)
        rep_local = verify_coloring(G, cand_compact, allowed_colors=list(range(used)))
        if verbose:
            print(f"  [AcceptCheck:{label}] feasible={rep_local['feasible']} used={used} conflicts={rep_local['num_conflicts']}")
        return bool(rep_local["feasible"]), cand_compact, used, rep_local

    stall_after_K = 8  # Trigger Step-5 termination when consecutive rounds without progress reach this value and K==ceil(zLP)
    
    # Schritt 0: LB + Initial solution (and compact color indices)
    clique_nodes = greedy_max_clique(G)
    LB = len(clique_nodes)
    init_col = smallest_last_coloring(G) if init_heuristic == "smallest_last" else dsatur_coloring(G)
    init_col, UB = compact_colors(init_col)   # Compact color indices to 0..UB-1
    best_UB_seen = UB
    best_time_sec = 0.0
    best_round = 0
    best_coloring = init_col
    K = UB  # Initialize LP/rounding palette K, later tightened by ceil(zLP)
    if verbose:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        is_big = (n >= 600) or (m * UB >= 2e7)
        # if is_big:
        #     restarts = min(restarts, 12)
        #     max_fix_per_round = min(max_fix_per_round, 10)
        #     strong_margin = max(strong_margin, 0.30)
        print(f"[Init] graph: |V|={n} |E|={m} |LB|={LB} |UB(init)|={UB} |K|={K} via {init_heuristic}")
    if enable_visualization:
        visualize_coloring(
            G, best_coloring, step="Init", round_id=0,
            out_dir=viz_out_dir, layout_seed=viz_layout_seed,
            only_colored=True, allowed_colors=list(range(UB)),
            conflict_nodes_fill_black=True,
        )
    if verbose:
        print(f"[Init] LB={LB} | UB(init:{init_heuristic})={UB}")
    if UB <= LB:
        rep = verify_coloring(G, best_coloring, allowed_colors=list(range(UB)))
        return dict(UB=UB, LB=LB, coloring=best_coloring, iters=0, log=[], stop_reason="UB==LB",
                    feasible=rep["feasible"], final_check=rep,best_time_sec=best_time_sec, best_round=best_round)

    # build LP once (initial K = UB)
    extra_cliques = top_maximal_cliques(G, max_cliques=50, min_size=max(4, LB+1), time_limit_sec=2.0)
    inc = IncrementalLP(    G, allowed_colors=list(range(UB)), clique_nodes=clique_nodes,
        extra_cliques=extra_cliques, add_precedence=True,
        edge_mode=edge_mode, lazy_threshold=lazy_threshold,)
    if verbose:
        print(f"[LP] model built once (incremental), initial |C|={len(inc.C)} (K={K}) with precedence & clique-cuts")

    K = UB  # palette size used for LP/rounding/verification; different from UB (best feasible coloring found)
    logs = []
    round_id = 0
    stop_reason = ""
    prev_UB = UB
    no_progress_rounds = 0
    stall_rounds = 0          # 连续无改进轮数
    last_improve_round = 0    # 上次改进发生在哪一轮（可选，用于日志）

        # ---------------- Big-graph strategy toggles ----------------
    # Goal: on big instances, solve the LP *as few times as possible* and reuse its fractional guidance.
    # Small instances keep the original behavior.
    BIG_ONE_SHOT_LP = True      # is_big -> only solve once (round 1), then reuse cached (zLP,x,y)
    BIG_SKIP_FIXING = True      # is_big -> skip FixPolicy/FixApply (avoids extra LP solves)
    BIG_DIVERSIFY_TIES = True   # is_big -> diversify DSATUR tie-breaking across restarts (no extra LP)

    lp_solves_total = 0
    lp_cache = None             # (zLP, x_frac, y_frac)
    lp_cache_round = 0
    lp_dirty = False
    lazy_added_since_solve = 0
    BIG_LAZY_RESOLVE_THRESHOLD = 2000     # 累计到多少条 conflict-cuts 再重解一次
    BIG_LAZY_RESOLVE_MIN_TLEFT = 120.0    # 剩余时间太少就别重解（避免尾部浪费）


    while time_left() > 0.0:
        t_round0 = time.monotonic()
        def big_mark(tag: str) -> None:
            if verbose and is_big:
                dt = time.monotonic() - t_round0
                print(f"  [BIG-T] {tag} dt={dt:.2f}s time_left={time_left():.1f}s")
        round_id += 1
        n = G.number_of_nodes()
        m = G.number_of_edges()
        est = m * K  # 用 K/|C| 更合理
        if verbose:
            print(f"[DBG-ENTER] round_id={round_id} elapsed={elapsed():.1f}s time_left={time_left():.1f}s UB={UB} K={K}")
        if budget_exhausted("round_begin"):
            break


        lazy = getattr(inc, "lazy_edges", False)

        # 更保守的 big 判定：不要 n>=600 就判 big
        is_big = lazy or (n >= 500) or (est >= 5e7)

        # 主 solve：大图也别太小；Round 1 给更充足时间
        if not is_big:
            lp_main_cap = 8.0
            lp_sub_cap  = 2.0
        else:
            lp_main_cap = 60.0
            lp_sub_cap  = 10.0 
            # if round_id <= 1:      # 第一次主 solve 更关键
            #     lp_main_cap = 300.0

        # 1) solve LP first (get zLP)  -- big-graph mode may reuse cached LP solution
        need_lp = (
            (not is_big)
            or (round_id == 1)
            or (lp_cache is None)
            or (not BIG_ONE_SHOT_LP)
            or (lp_dirty and lazy_added_since_solve >= BIG_LAZY_RESOLVE_THRESHOLD and time_left() >= BIG_LAZY_RESOLVE_MIN_TLEFT)
        )


        if (is_big and (not need_lp) and lp_cache is not None):
            zLP, x_frac, y_frac = lp_cache
            if verbose:
                age = round_id - lp_cache_round
                print(f"  [BIG-LP] reuse cached LP from round {lp_cache_round} (age={age} rounds, lp_solves_total={lp_solves_total})")
        else:
            inc.set_time_limit_ms(int(max(0.05, min(time_left(), lp_main_cap)) * 1000))
            print(f"[DBG] n={n} m={m} UB={UB} K={K} lazy={getattr(inc,'lazy_edges',False)} "
                  f"is_big={is_big} lp_main_cap={lp_main_cap}")

            try:
                info = inc.solve()
                big_mark("after_lp_or_cache")

                lp_solves_total += 1
                if verbose and is_big:
                    print(f"  [BIG-LP] solve_count={lp_solves_total} (main solve)")
            except TimeoutError:
                if budget_exhausted("lp_solve_timeout"):
                    break
                stop_reason = "lp_timeout"
                if verbose:
                    print(f"  [LP] solve timed out under per-call cap={lp_main_cap}s, time_left={time_left():.1f}s")

                if lp_cache is not None:
                    zLP, x_frac, y_frac = lp_cache
                    if verbose:
                        print("  [BIG-LP] fallback to cached LP due to timeout")
                else:
                    # Fallback: pseudo guidance from current best coloring (no LP info)
                    x_frac = {(v, best_coloring.get(v, 0)): 1.0 for v in G.nodes()}
                    y_frac = {c: 1.0 for c in range(K)}
                    zLP = float(LB)
                    if verbose:
                        print("  [BIG-LP] no cached LP available -> using heuristic pseudo-guidance (x from best_coloring)")
            else:
                zLP, x_frac, y_frac = info["z_LP"], info["x_frac"], info["y_frac"]
                if is_big and lp_dirty:
                    if verbose:
                        print(f"  [BIG-LAZY] re-solved LP after dirty; reset counters (added={lazy_added_since_solve})")
                    lp_dirty = False
                    lazy_added_since_solve = 0

                # LAZY separation: big graphs do at most 1 round + 1 short re-solve
                if getattr(inc, "lazy_edges", False):
                    sep_rounds = 1 if is_big else 5
                    for _ in range(sep_rounds):
                        if time_left() <= 0.0:
                            break
                        added = inc.lazy_separate_from_lp(x_frac, y_frac, top_t=10, eps=1e-6, max_new=100000)
                        if verbose:
                            print(f"  [LazyEdge] added={added}")
                        if added <= 0:
                            break
                        re_cap = 0.5 if is_big else 0.3
                        inc.set_time_limit_ms(int(max(0.05, min(time_left(), re_cap)) * 1000))
                        try:
                            info2 = inc.solve()
                            big_mark("after_lp_or_cache")

                            lp_solves_total += 1
                            if verbose and is_big:
                                print(f"  [BIG-LP] solve_count={lp_solves_total} (lazy re-solve)")
                        except TimeoutError:
                            break
                        zLP, x_frac, y_frac = info2["z_LP"], info2["x_frac"], info2["y_frac"]

                if is_big and BIG_ONE_SHOT_LP:
                    lp_cache = (zLP, x_frac, y_frac)
                    lp_cache_round = round_id
                    if verbose:
                        print(f"  [BIG-LP] cached LP at round {round_id} (zLP={zLP:.3f})")

    #     # 1) solve LP first (get zLP)
    #     inc.set_time_limit_ms(int(max(0.05, min(time_left(), lp_main_cap)) * 1000))
    #     print(f"[DBG] n={n} m={m} UB={UB} K={K} lazy={getattr(inc,'lazy_edges',False)} "
    #   f"is_big={is_big} lp_main_cap={lp_main_cap}")

    #     try:
    #         info = inc.solve()
    #     except TimeoutError:
    #         # 这是“LP 单次 solve 被 SetTimeLimit 截断”，不是全局 8h 用完
    #         if budget_exhausted("lp_solve_timeout"):
    #             break  # 只有全局时间真没了才退出

    #         stop_reason = "lp_timeout"  # 改名，避免误导
    #         if verbose:
    #             print(f"  [LP] solve timed out under per-call cap={lp_main_cap}s, time_left={time_left():.1f}s -> continue")

    #         # 策略 A：提高本轮 LP cap 再试一次（推荐）
    #         lp_main_cap = min(lp_main_cap * 2.0, 120.0)  # 上限可调，比如 120s
    #         inc.set_time_limit_ms(int(max(0.05, min(time_left(), lp_main_cap)) * 1000))
    #         continue  # 不退出外层 while，继续下一轮/重试

    #     zLP, x_frac, y_frac = info["z_LP"], info["x_frac"], info["y_frac"]
    #     # LAZY: only after we have a current LP solution (x_frac/y_frac exist)
    #     if getattr(inc, "lazy_edges", False):
    #         sep_rounds = 5
    #         for _ in range(sep_rounds):
    #             if time_left() <= 0.0:
    #                 break
    #             added = inc.lazy_separate_from_lp(x_frac, y_frac, top_t=10, eps=1e-6, max_new=100000)
    #             if verbose:
    #                 print(f"  [LazyEdge] added={added}")
    #             if added <= 0:
    #                 break
    #             # short re-solve after adding cuts
    #             inc.set_time_limit_ms(int(max(0.05, min(time_left(), 0.3)) * 1000))
    #             try:
    #                 info2 = inc.solve()
    #             except TimeoutError:
    #                 break
    #             zLP, x_frac, y_frac = info2["z_LP"], info2["x_frac"], info2["y_frac"]

        if verbose:
            ceilK = int(math.ceil(zLP - 1e-12))
            t_sec = elapsed()
            print(f"[DBG-TL-round] round_id={round_id} time_left={time_left():.1f}s")

            print(f"[Round {round_id}] zLP={zLP:.6f} | ceil(zLP)={ceilK} | K={K} | UB={UB} | t={t_sec:.2f}s")
        # # 2) decide whether to tighten K based on ceil(zLP) (safe: try - rollback if fails)
        # n = G.number_of_nodes()
        # m = G.number_of_edges()
        # is_big = (n >= 600) or (m * K >= 2e7)
        # lb_lp = int(math.ceil(zLP - 1e-12))   # 统一先算出来，避免未定义
        # lazy = getattr(inc, "lazy_edges", False)
        # if (not lazy) and is_big:
        #     K_target = K 
        # if lazy:
        #     # lazy 模式：只有当 LP 下界已经非常接近 UB 时才可信，才允许 shrink
        #     if lb_lp >= UB - 3:
        #         K_target = max(LB, lb_lp)
        #     else:
        #         K_target = K   # 不 shrink，避免 K 被错误压到 clique 附近

        # else:
        #     K_target = max(LB, int(math.ceil(zLP - 1e-12)), UB - 1, lb_lp)  # keep on small graphs
        lb_lp = int(math.ceil(zLP - 1e-12))

        # 关键：K 不要一次性追 lb_lp；用一个“窗口”控制收缩幅度
        WINDOW_SMALL = 2   # 小图：更激进（更精细）
        WINDOW_BIG   = 6   # 大图：更保守（保证 rounding 有可行空间）

        lazy = bool(getattr(inc, "lazy_edges", False))
        n = G.number_of_nodes()
        m = G.number_of_edges()
        is_big = lazy or (n >= 500) or (m * K >= 2e6)   # 注意这里用 K，不用 UB

        W = WINDOW_BIG if is_big else WINDOW_SMALL

        # 目标：让 K 至少不要比 (lb_lp + W) 更小；否则 rounding 基本必死
        K_floor = max(LB, lb_lp + W)

        # 同时：K 也不应该超过当前 UB（否则浪费）
        K_ceiling = UB

        # 最终目标 K：在 [K_floor, K_ceiling] 之间，且不要上调 K
        K_target = min(K, max(K_floor, K_ceiling))   # 注意：max(...)=K_ceiling 若 K_ceiling>=K_floor
        K_target = min(K_target, UB)                 # 再保险一次

        # 如果 floor 已经超过 UB，说明 LP 下界已经非常接近 UB（或窗口设太大）
        # 这种情况下就直接让 K 对齐 UB（别乱 shrink）
        if K_floor > UB:
            K_target = UB
        K_shrunk_this_round = False
        x_fix_applied_this_round = False
        improved_this_round = False

        if (not is_big) and (K_target < K):
            tok = [inc.lock_prefix_K(K_target)]
            sub_cap = 50
            inc.set_time_limit_ms(int(max(0.05, min(time_left(), sub_cap)) * 1000))
            try:    
                ok_info, ok = inc.try_apply_and_solve(tok)
            except TimeoutError:
                ok_info, ok = None, False
                if verbose:
                    print("  [Fix] shrinkK solve timed out - rollback and continue (keep current K)")
            oldK = K
            if ok:
                K = K_target
                K_shrunk_this_round = True
                if verbose:
                    print(f"  [Fix] shrink K: {oldK} -> {K_target} (lb_lp={lb_lp}, UB={UB})")
                dbg = _debug_candidate_from_xfrac(ok_info["x_frac"], K_eff=K)
                if enable_visualization:
                    viz("ShrinkK-ok", round_id, dbg, K, only_colored=False)
                # continue with updated solution to avoid re-solving
                zLP, x_frac, y_frac = ok_info["z_LP"], ok_info["x_frac"], ok_info["y_frac"]
            else:
                inc.revert_all(tok)
                if verbose:
                    print(f"  [Fix] shrink to K={K_target} infeasible - rollback; keep K={K}")
                dbg = _debug_candidate_from_xfrac(x_frac, K_eff=K_target)
                if enable_visualization:
                    viz("ShrinkK-rollback", round_id, dbg, K_target, only_colored=False)
        if is_big and (K_target < K) and verbose:
            print(f"  [BIG] skip shrinkK via LP (K={K}, K_target={K_target})")

        # 3) rounding with K colors
        if budget_exhausted("before_rounding"):
            break
        if is_big:
            vrcl = 1
            crcl = 3
            xjit = 0.0
            tau  = 0.02
        else:
            vrcl = 1
            crcl = 1
            xjit = 0.0
            tau  = 0.0
        # --- BIG: per-round budget to prevent RNR from consuming whole time limit ---
        global_deadline_ts = deadline  # 你原来用于全局 time limit 的那个（必须是 monotonic 时间戳）
        if is_big:
            # 让每轮最多用掉剩余时间的一部分，保证还能进 local search / UB1 / 下一轮
            time_left_now = global_deadline_ts - time.monotonic()
            round_budget = max(20.0, min(90.0, 0.20 * time_left_now))  # 20~90s 或剩余时间 20%
            round_deadline_ts = min(global_deadline_ts, time.monotonic() + round_budget)
            if verbose:
                print(f"  [BIG-BUDGET] round_budget={round_budget:.1f}s")
        else:
            round_deadline_ts = global_deadline_ts
        cand = round_and_repair_multi(G, x_frac, y_frac, current_UB=K,
                                        restarts=restarts, seed=algo_seed + round_id, perturb_y=perturb_y,deadline_ts=round_deadline_ts,
                                        diversify_ties=(is_big and BIG_DIVERSIFY_TIES),
                                        debug=(is_big and verbose),
                                        debug_tag=f"RNR-r{round_id}",
                                        vertex_rcl=vrcl, color_rcl=crcl, x_jitter=xjit,color_tau=tau)
        if cand is None:
            if verbose:
                print("  [RNR] returned None (deadline hit) -> fallback to incumbent")
            cand = best_coloring  # 这是你始终维护的 incumbent feasible dict
        big_mark("after_rounding_multi")

        rep = verify_coloring(G, cand, allowed_colors=list(range(K)))
        if budget_exhausted("after_rounding"): break
        if enable_visualization:
            viz("Rounding", round_id, cand, K)
        if verbose:
            used_try = len(set(cand.values()))
            print(f"  [Rounding] feasible={rep['feasible']} | used={used_try} | conflicts={rep['num_conflicts']}")

        if rep["feasible"]:
            # (A) feasible path: maybe improve UB, sync K, etc.
            cand, used = compact_colors(cand)
            cand2, reduced = consolidate_colors(G, dict(cand), passes=5)
            if reduced:
                cand2, used = compact_colors(cand2)
                cand = cand2
                if enable_visualization:
                    viz("Local-Consolidate", round_id, cand, max(K, used))
            if used < UB:
                UB = used
                best_coloring = cand
                improved_this_round = True
                
                if verbose:
                    print(f"  [Rounding/Local] improved UB -> {UB}")
                if K > UB:
                    if is_big:
                        big_mark("after_local_search")

                        K = UB
                        mark_best(UB, round_id)
                        if verbose:
                            print(f"  [BIG-Sync] K aligned to new UB without LP: K={K}")
                    else:        
                        tok2 = [inc.lock_prefix_K(UB)]
                        sub_cap = 50
                        inc.set_time_limit_ms(int(max(0.05, min(time_left(), sub_cap)) * 1000))
                        try:
                            ok_info2, ok2 = inc.try_apply_and_solve(tok2)
                        except TimeoutError:
                            ok_info2, ok2 = None, False
                            if verbose:
                                print("  [rounding K] solve timed out - rollback and continue")
                        if ok2:
                            K = UB
                            if verbose:
                                print(f"  [Sync] K aligned to new UB: K={K}")
                            zLP, x_frac, y_frac = ok_info2["z_LP"], ok_info2["x_frac"], ok_info2["y_frac"]
                            mark_best(UB, round_id)
                        else:
                            inc.revert_all(tok2)
                if UB <= LB:
                    stop_reason = "UB==LB"
                    break
        else:
            # (B) infeasible path: try K+1 side rounding before any fixing
            if inc.lazy_edges:
                added_conf = inc.lazy_add_conflict_cuts(cand, max_new=50000)
                if verbose:
                    print(f"  [LazyEdge-Conflicts] added={added_conf}")
                # BIG: one-shot LP mode -> do NOT re-solve LP here
                if is_big and BIG_ONE_SHOT_LP:
                    if verbose and added_conf > 0:
                        print("  [BIG] conflict cuts added, skip LP re-solve (one-shot LP mode)")
                else:
                    if added_conf > 0:
                        inc.set_time_limit_ms(int(max(0.05, min(time_left(), lp_sub_cap)) * 1000))
                        try:
                            info = inc.solve()
                            big_mark("after_lp_or_cache")
                            lp_solves_total += 1
                            if verbose and is_big:
                                print(f"  [BIG-LP] solve_count={lp_solves_total} (conflict re-solve)")
                            zLP, x_frac, y_frac = info["z_LP"], info["x_frac"], info["y_frac"]
                        except TimeoutError:
                            pass

            if verbose:
                print(f"  [Side] trying K+1 rounding (K={K} - {K+1}) to break stalemate")
            newUB, newCol, applied = try_side_rounding(
                    G, x_frac, y_frac, K, UB,
                    restarts=restarts, perturb_y=perturb_y, seed=algo_seed+ 9000 + round_id, verbose=verbose,
                    enable_visualization=enable_visualization,
                    viz_round_id=round_id, viz_out_dir=viz_out_dir, viz_layout_seed=viz_layout_seed,
                    step_name="Side-K+1",
                )
            if applied and newUB < UB:

                # UB = newUB
                # best_coloring = newCol
                # improved_this_round = True
                ok, cand_ok, used_ok, rep_ok = _accept_if_feasible("LP-Pack", newCol, newUB)
                if ok and used_ok < UB:
                    UB = used_ok
                    best_coloring = cand_ok
                    improved_this_round = True
                mark_best(UB, round_id)
                if UB <= LB:
                    stop_reason = "UB==LB"
                    break

        if verbose:
            used_try = len(set(cand.values()))
            print(f"  [Rounding] feasible={rep['feasible']} | used={used_try} | conflicts={rep['num_conflicts']}")
            # ---------------- BIG: infeasible fallback (do NOT terminate) ----------------
            if is_big and (not rep["feasible"]):
                if verbose:
                    print(f"  [BIG-RNR] infeasible candidate -> conflicts={rep['num_conflicts']} (will fallback)")

                # 1) feed conflicts into lazy LP (but do NOT re-solve immediately)
                if getattr(inc, "lazy_edges", False):
                    added_conf = inc.lazy_add_conflict_cuts(cand, max_new=50000)
                    if added_conf > 0:
                        lp_dirty = True
                        lazy_added_since_solve += added_conf
                    if verbose:
                        print(f"  [BIG-LazyEdge-Conflicts] added={added_conf} "
                            f"dirty={lp_dirty} lazy_added_since_solve={lazy_added_since_solve}")

                # 2) quick "safe" rounding retry with conservative settings (cheap, no LP solve)
                #    goal: at least get a feasible coloring, not necessarily improve UB
                safe_budget = min(10.0, max(2.0, 0.02 * time_left()))  # 2~10s
                safe_deadline = time.monotonic() + safe_budget
                if verbose:
                    print(f"  [BIG-RNR] safe-retry budget={safe_budget:.1f}s")

                cand2 = round_and_repair_multi(
                    G, x_frac, y_frac, current_UB=K,
                    restarts=min(3, restarts),                   # 很小的重启数
                    seed=algo_seed + 777777 + round_id,
                    perturb_y=perturb_y,
                    deadline_ts=safe_deadline,
                    diversify_ties=False,
                    debug=False,
                    vertex_rcl=1, color_rcl=1, x_jitter=0.0,     # 保守：接近原始 deterministic
                )
                rep2 = verify_coloring(G, cand2, allowed_colors=list(range(K)))

                if rep2["feasible"]:
                    cand = cand2
                    rep = rep2
                    if verbose:
                        print("  [BIG-RNR] safe-retry SUCCESS -> proceed with feasible cand")
                else:
                    # 3) final fallback: revert to incumbent feasible coloring
                    cand = best_coloring
                    rep = verify_coloring(G, cand, allowed_colors=list(range(K)))
                    if verbose:
                        print(f"  [BIG-RNR] fallback to incumbent -> feasible={rep['feasible']} conflicts={rep['num_conflicts']}")
            # ---------------- end BIG fallback ----------------

        if not is_big:    
            # 3.5) UB-1 side attempt (allow K+1 colors, only for constructing better feasible solutions; doesn't change LP locking)
            newUB, newCol, applied = try_side_rounding(
                    G, x_frac, y_frac, K, UB,
                    restarts=restarts, perturb_y=perturb_y, seed=algo_seed+9000 + round_id, verbose=verbose,
                    enable_visualization=enable_visualization,
                    viz_round_id=round_id, viz_out_dir=viz_out_dir, viz_layout_seed=viz_layout_seed,
                    step_name="UB-1-Side",
                )
            if applied and newUB < UB:
                ok, cand_ok, used_ok, rep_ok = _accept_if_feasible("LP-Pack", newCol, newUB)
                if ok and used_ok < UB:
                    UB = used_ok
                    best_coloring = cand_ok
                    improved_this_round = True
                mark_best(UB, round_id)
                if UB <= LB:
                    stop_reason = "UB==LB"
                    break
            #  and LP-guided pack: when UB == K+1, pack highest color layer back into 0..K-1
            if verbose and UB == K + 1:
                print(f"  [PackWindow] UB==K+1 (UB={UB}, K={K}) - attempt LP-guided pack and UB-1 greedy")
        if (not is_big) and (UB == K + 1):
            if UB == K + 1:
                # B1) LP-guided highest color layer packing
                newUB, newCol, applied = try_lp_guided_pack(G, best_coloring, x_frac, K, UB, verbose=verbose,
                    enable_visualization=enable_visualization,
                    viz_round_id=round_id, viz_out_dir=viz_out_dir, viz_layout_seed=viz_layout_seed,)
                if applied and newUB < UB:
                    ok, cand_ok, used_ok, rep_ok = _accept_if_feasible("LP-Pack", newCol, newUB)
                    if ok and used_ok < UB:
                        UB = used_ok
                        best_coloring = cand_ok
                        improved_this_round = True
                    
                    if K > UB:
                        tok3 = [inc.lock_prefix_K(UB)]
                        sub_cap = 50
                        inc.set_time_limit_ms(int(max(0.05, min(time_left(), sub_cap)) * 1000))
                        try:
                            ok_info3, ok3 = inc.try_apply_and_solve(tok3)
                        except TimeoutError:
                            ok_info3, ok3 = None, False
                            if verbose:
                                print("  [LP-Pack] solve timed out - rollback and continue")
                        if ok3:
                            K = UB
                            mark_best(UB, round_id)
                            if verbose:
                                print(f"  [Sync] K aligned to new UB: K={K}")
                        else:
                            inc.revert_all(tok3)
                    if UB <= LB:
                        stop_reason = "UB==LB"
                        break
                # B2) pure graph-domain UB-1 greedy 
                newUB, newCol, applied = try_graph_ub1_greedy(G, best_coloring, UB, verbose=verbose,
                    enable_visualization=enable_visualization,
                    viz_round_id=round_id, viz_out_dir=viz_out_dir, viz_layout_seed=viz_layout_seed,)

                if applied and newUB < UB:
                    ok, cand_ok, used_ok, rep_ok = _accept_if_feasible("LP-Pack", newCol, newUB)
                    if ok and used_ok < UB:
                        UB = used_ok
                        best_coloring = cand_ok
                        improved_this_round = True
                    
                    if K > UB:
                        tok4 = [inc.lock_prefix_K(UB)]
                        sub_cap = 50
                        inc.set_time_limit_ms(int(max(0.05, min(time_left(), sub_cap)) * 1000))
                        try:
                            ok_info4, ok4 = inc.try_apply_and_solve(tok4)
                        except TimeoutError:
                            ok_info4, ok4 = None, False
                            if verbose:
                                print("  [UB-1 Greedy] solve timed out - rollback and continue")
                        if ok4:
                            K = UB
                            mark_best(UB, round_id)
                            if verbose:
                                print(f"  [Sync] K aligned to new UB: K={K}")
                        else:
                            inc.revert_all(tok4)
                    if UB <= LB:
                        stop_reason = "UB==LB"
                        break

        # ---- BIG: UB-1 squeezing phase (NO LP SOLVE) ----
        # 只在“确实接近可降 1 色”的时候做，避免前期 UB 很大时白耗时
        UB1_STALL = 2          # 连续 2 轮无改进就触发一次
        UB1_EVERY = 3          # 或者每 3 轮强制触发一次（防止一直不触发）
        UB1_MIN_LEFT = 60.0    # 剩余时间太少就别折腾

        if is_big and (x_frac is not None) and (best_coloring is not None) \
            and (time_left() > UB1_MIN_LEFT) \
            and (stall_rounds >= UB1_STALL or (round_id % UB1_EVERY == 0)):


            time_left_now = time_left()  # 你上面定义过的 closure
            squeeze_budget = min(30.0, max(5.0, 0.10 * time_left_now))


            max_dec = 3
            max_tries = 12
            dec_done = 0

            print(f"  [BIG-UB1] start: UB={UB} LB={LB} budget={squeeze_budget:.1f}s")
            quick_success = False
            squeeze_t0  = time.monotonic()   # 用于统计 squeezing 总耗时（顺便把你后面那个 t0 挪到这里）

            # (0) quick UB-1 rounding attempt (cheap)
            K_try = UB - 1
            if K_try > LB + 1:
                quick_budget = min(12.0, max(3.0, 0.03 * time_left()))
                quick_deadline = min(deadline, time.monotonic() + quick_budget)

                cand_try = round_and_repair_multi(
                    G, x_frac, y_frac,
                    current_UB=K_try,
                    restarts=min(8, restarts),
                    seed=algo_seed + 123456 + round_id,
                    perturb_y=perturb_y,
                    deadline_ts=quick_deadline,
                    diversify_ties=True,
                    debug=False,
                    debug_tag=f"UB1-RNR-r{round_id}",
                    vertex_rcl=1, color_rcl=3, x_jitter=0.0, color_tau=0.02,
                )
            if cand_try is None:
                if verbose:
                    print("  [BIG-UB1] quick RNR returned None (deadline hit) -> skip")
            else:
                rep_try = verify_coloring(G, cand_try, allowed_colors=list(range(K_try)))
                if rep_try["feasible"]:
                    ok, cand_ok, used_ok, rep_ok = _accept_if_feasible("BIG-UB1-RNR", cand_try, K_try)
                    if ok and used_ok < UB:
                        UB = used_ok
                        best_coloring = cand_ok
                        improved_this_round = True
                        dec_done += 1
                        if K > UB:
                            K = UB
                        mark_best(UB, round_id)
                        print(f"  [BIG-UB1] SUCCESS via quick RNR: new UB={UB}")
                        quick_success = True

            tries_done = 0

            if not quick_success:
                for tries in range(max_tries):
                    tries_done = tries + 1

                    # 预算 / 停止条件
                    if dec_done >= max_dec:
                        break
                    if UB <= LB + 1:
                        break
                    if (time.monotonic() - t0) >= squeeze_budget:
                        break

                    K_try = UB - 1
                    improved = False

                    if verbose:
                        print(f"  [BIG-UB1] try#{tries+1}: LP-pack UB {UB} -> {K_try}")

                    # (1) LP-pack
                    ub2, col2, applied = try_lp_guided_pack(G, best_coloring, x_frac, K_try, UB, verbose=verbose)
                    if applied and (col2 is not None):
                        ok, cand_ok, used_ok, rep_ok = _accept_if_feasible("BIG-UB1-LP-Pack", col2, ub2)
                        if ok and used_ok < UB:
                            UB = used_ok
                            best_coloring = cand_ok
                            improved_this_round = True
                            dec_done += 1
                            improved = True
                            if K > UB:
                                K = UB
                            mark_best(UB, round_id)
                            if verbose:
                                print(f"  [BIG-UB1] SUCCESS via LP-pack: new UB={UB} (dec_done={dec_done})")

                    # (2) UB1-greedy（只在 pack 没成功时才做）
                    if (not improved) and ((time.monotonic() - t0) < squeeze_budget):
                        if verbose:
                            print(f"  [BIG-UB1] try#{tries+1}: UB1-greedy on UB={UB}")

                        ub3, col3, applied3 = try_graph_ub1_greedy(G, best_coloring, UB, verbose=verbose)
                        if applied3 and (col3 is not None):
                            ok, cand_ok, used_ok, rep_ok = _accept_if_feasible("BIG-UB1-UB1Greedy", col3, ub3)
                            if ok and used_ok < UB:
                                UB = used_ok
                                best_coloring = cand_ok
                                improved_this_round = True
                                dec_done += 1
                                improved = True
                                if K > UB:
                                    K = UB
                                mark_best(UB, round_id)
                                if verbose:
                                    print(f"  [BIG-UB1] SUCCESS via UB1-greedy: new UB={UB} (dec_done={dec_done})")

                    # 两个算子都没改进：直接退出 squeezing（避免白耗时）
                    if not improved:
                        if verbose:
                            print("  [BIG-UB1] no improvement, stop squeezing.")
                        break

            print(f"  [BIG-UB1] done: dec_done={dec_done} tries={tries_done} spent={time.monotonic()-squeeze_t0:.1f}s UB={UB}")

        # ---- end BIG UB-1 squeezing ----

        # Big-graph mode: skip FixPolicy/FixApply (each would trigger additional LP solves).
        if is_big and BIG_SKIP_FIXING:
            if verbose:
                print("  [BIG] skip FixPolicy/FixApply to avoid extra LP solves")
        else:
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
                sub_cap = 50
                inc.set_time_limit_ms(int(max(0.05, min(time_left(), sub_cap)) * 1000))
                try:
                    ok_info3, ok3 = inc.try_apply_and_solve(tokens)
                except TimeoutError:
                    ok_info3, ok3 = None, False
                    if verbose:
                        print("  [FixApply] solve timed out - rollback and continue")
                if verbose:
                    print(f"  [FixApply] status={'OK' if ok3 else 'ROLLBACK'}")
                if not ok3:
                    inc.revert_all(tokens)
                    if enable_visualization:
                        dbg = _debug_candidate_from_xfrac(x_frac, K_eff=K)
                        viz("FixApply-rollback", round_id, dbg, K, only_colored=False)
                else:
                    x_fix_applied_this_round = True
                    if budget_exhausted("before_rounding"):
                        break
                    cand3 = round_and_repair_multi(G, ok_info3["x_frac"], ok_info3["y_frac"],
                                                current_UB=K, restarts=max(2, restarts // 2),
                                                seed=algo_seed+7777 + round_id, perturb_y=perturb_y,deadline_ts=deadline)
                    rep3 = verify_coloring(G, cand3, allowed_colors=list(range(K)))
                    if budget_exhausted("after_rounding"): break
                    if enable_visualization:
                        viz("After-Fix-Rounding", round_id, cand3, K, only_colored=rep3["feasible"])
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
                                sub_cap = 50
                                inc.set_time_limit_ms(int(max(0.05, min(time_left(), sub_cap)) * 1000))
                                try:
                                    ok_info4, ok4 = inc.try_apply_and_solve(tok4)
                                except TimeoutError:
                                    ok_info3, ok3 = None, False
                                    if verbose:
                                        print("  [FixApply] solve timed out - rollback and continue")

                                if ok4:
                                    K = UB
                                    mark_best(UB, round_id)
                                    if verbose:
                                        print(f"  [Sync] K aligned to new UB: K={K}")
                                else:
                                    inc.revert_all(tok4)
                            if UB <= LB:
                                stop_reason = "UB==LB"
                                break
        if verbose:
            print(f"[DBG-TL-round] round_id={round_id} time_left={time_left():.1f}s")

            print(f"  [Round {round_id} END] UB={UB} | K={K} | ceil(zLP)={int(math.ceil(zLP - 1e-12))} "
                f"| improved={improved_this_round} | fixed={(K_shrunk_this_round or x_fix_applied_this_round)}")
        if enable_visualization:
            viz("End-Of-Round", round_id, best_coloring, UB)
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
        if improved_this_round:
            stall_rounds = 0
            last_improve_round = round_id
        else:
            stall_rounds += 1

        if verbose:
            print(f"  [STALL] stall_rounds={stall_rounds} last_improve_round={last_improve_round}")

        if is_big:
            print(f"  [BIG] lp_solves_total={lp_solves_total} lp_cache_round={lp_cache_round}")
            big_mark("round_end")



    if not stop_reason:
        stop_reason = "time_limit" if time_left() <= 0.0 else "stopped_without_reason"
    
    if verbose:
        print(f"[DBG-EXIT] elapsed={elapsed():.1f}s time_left={time_left():.1f}s stop_reason={stop_reason}")

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
    if verbose and is_big:
            print(f"  [BIG] lp_solves_total={lp_solves_total}")

    if enable_visualization:
        try:
            visualize_coloring(
                G, best_coloring, step="Final", round_id=round_id,
                out_dir=viz_out_dir, layout_seed=viz_layout_seed,
                only_colored=True,
                allowed_colors=None,
                conflict_nodes_fill_black=True,
            )
        except Exception as e:
            if verbose:
                print(f"  [Viz] skip (Final) due to: {e}")
    return dict(
        UB=UB, LB=LB, coloring=best_coloring, iters=len(logs), log=logs,
        stop_reason=stop_reason, feasible=final_report["feasible"], final_check=final_report,
        best_time_sec=best_time_sec, best_round=best_round
    )


# def _bootstrap_initial_from_lp(
#     G,
#     headroom: int = 3,
#     max_k_increase: int = None,
#     restarts: int = 16,
#     time_limit_bootstrap: float = 10.0,
#     verbose: bool = True,
# ) -> Tuple[int, int, List[int], List[List[int]], Dict[int, int], float]:
#     """
#     use LP relaxation + rounding/repair to obtain the FIRST feasible coloring as the initial UB.
#     - start from LB = |clique| and set K = LB + headroom;
#     - iff rounding fails, increase K and retry (until success or limits).
#     returns: (UB, LB, clique_nodes, extra_cliques, best_coloring, z_LP_at_success)
#     """
#     t0 = time.monotonic()
#     stop_reason = ""
#     deadline = t0 + float(time_limit_sec)
#     def elapsed():
#         return time.monotonic() - t0

#     def time_left():
#         return deadline - time.monotonic()
#     n = G.number_of_nodes()
#     def budget_exhausted(stage: str) -> bool:
#         nonlocal stop_reason
#         if time_left() <= 0.0:
#             stop_reason = "time_limit"
#             if verbose:
#                 print(f"[Time] budget exhausted at {stage}, UB={UB if UB is not None else 'N/A'}")
#             return True
#         return False
#     clique_nodes = greedy_max_clique(G)
#     LB = len(clique_nodes)

#     # clique cuts to strengthen the LP
#     extra_cliques = top_maximal_cliques(G, max_cliques=50, min_size=max(4, LB + 1), time_limit_sec=2.0)

#     # start with K = LB + headroom and increase if rounding fails
#     K = min(n, max(LB, LB + headroom))
#     if max_k_increase is None:
#         max_k_increase = max(0, n - K)

#     last_zLP = None
#     for bump in range(max_k_increase + 1):
#         if time.time() - t0 > time_limit_bootstrap:
#             break

#         allowed_colors = list(range(K))
#         # Build and solve the LP
#         model, var_maps = build_lp_model(
#             G=G,
#             allowed_colors=allowed_colors,
#             clique_nodes=clique_nodes,
#             extra_cliques=extra_cliques,
#             add_precedence=True,
#         )
#         info = solve_lp_and_extract(model, var_maps)
#         last_zLP = info["z_LP"]

#         # multi-start rounding + repair
#         if budget_exhausted("before_rounding"):
#             break
#         cand = round_and_repair_multi(G, info["x_frac"], info["y_frac"], current_UB=K, restarts=restarts, seed=bump,deadline_ts=deadline)
#         rep = verify_coloring(G, cand, allowed_colors=list(range(K)))
#         if budget_exhausted("after_rounding"): break
#         if verbose:
#             print(f"[Init-LP] try K={K} -> feasible={rep['feasible']} | used={len(set(cand.values()))} | z_LP={last_zLP:.6f}")
#             if not rep["feasible"]:
#                 print(f"[Init-LP] conflicts_sample={rep['conflicts_sample'][:10]}")

#         if rep["feasible"]:
#             UB0 = len(set(cand.values()))
#             return UB0, LB, clique_nodes, extra_cliques, cand, last_zLP

#         # otherwise increase K and try once again
#         K = min(n, K + 1)

#     # fallback: try K = n one last time (should almost always succeed)
#     if K < n:
#         allowed_colors = list(range(n))
#         model, var_maps = build_lp_model(
#             G=G, allowed_colors=allowed_colors, clique_nodes=clique_nodes,
#             extra_cliques=extra_cliques, add_precedence=True
#         )
#         info = solve_lp_and_extract(model, var_maps)
#         last_zLP = info["z_LP"]

#         if budget_exhausted("before_rounding"):break
#         cand = round_and_repair_multi(G, info["x_frac"], info["y_frac"], current_UB=n, restarts=restarts, seed=999,deadline_ts=deadline)
#         rep = verify_coloring(G, cand, allowed_colors=allowed_colors)
#         if budget_exhausted("after_rounding"): break
#         if verbose:
#             print(f"[Init-LP] fallback K=n -> feasible={rep['feasible']} | used={len(set(cand.values()))} | z_LP={last_zLP:.6f}")
#         if rep["feasible"]:
#             UB0 = len(set(cand.values()))
#             return UB0, LB, clique_nodes, extra_cliques, cand, last_zLP

#     #maybe not happen: return an empty coloring (will be caught by assertions later)
#     return n, LB, clique_nodes, extra_cliques, {}, (last_zLP if last_zLP is not None else float("inf"))


# def try_shrink_UB_by_one_LP(
#     G, UB: int, clique_nodes: List[int], extra_cliques: List[List[int]],
#     seed: int = 0, verbose: bool = False,
#     enable_visualization: bool = False, viz_round_id: int = 0,
#     viz_out_dir: str = "visualisierung", viz_layout_seed: int = 42
# ):
#     """LP with K=UB-1 + multi-start rounding; success -> return (UB-1, coloring, True)."""
#     if UB <= len(clique_nodes):
#         # 没法再降，也给一张 -nc 占位图（可选）
#         return UB, None, False

#     K2 = list(range(UB - 1))
#     model2, var_maps2 = build_lp_model(
#         G, allowed_colors=K2, clique_nodes=clique_nodes,
#         extra_cliques=extra_cliques, add_precedence=True
#     )
#     info2 = solve_lp_and_extract(model2, var_maps2)
#     if budget_exhausted("before_rounding"):
#         break
#     cand = round_and_repair_multi(G, info2["x_frac"], info2["y_frac"],
#                                   current_UB=UB - 1, restarts=8, seed=seed,deadline_ts=deadline)
#     rep = verify_coloring(G, cand, allowed_colors=K2)
#     if budget_exhausted("after_rounding"): break
#     # —— 成败都出图（cand 可能是 infeasible 的失败候选，我们一样画）
#     if enable_visualization:
#         visualize_coloring(
#             G, cand if cand else {},
#             step=("LP-UB-1" if rep["feasible"] else "LP-UB-1-failed"),
#             round_id=viz_round_id,
#             out_dir=viz_out_dir, layout_seed=viz_layout_seed,
#             only_colored=True,
#             allowed_colors=K2,
#             conflict_nodes_fill_black=True,
#         )

#     if rep["feasible"]:
#         if verbose:
#             print(f"  [UB-1 Test (LP)] success -> UB={UB-1}")
#         return UB - 1, cand, True
#     else:
#         if verbose:
#             print(f"  [UB-1 Test (LP)] failed")
#         return UB, None, False

# def run_iterative_lp(
#     G,
#     time_limit_sec: int = 60,
#     max_rounds: int = 200,
#     stall_rounds: int = 10,
#     min_rounds: int = 5,
#     verbose: bool = True,
# ) -> Dict[str, Any]:
#     """
#     iterative LP-based heuristic (first feasible solution from LP rounding):
#     - bootstrap: start from LB; build LP with K=LB+headroom and use rounding/repair for initial feasible UB0.
#     - iteration: each round runs LP - rounding -> conservative fixing (K=0..UB-1) - local search - UB-1 (graph) - UB-1 (LP).
#     - stop when UB==LB, or stalled, or time/round limits are hit.
#     """
#     t_all0 = time.time()

#     #Bootstrap: use LP rounding to get the first feasible coloring
#     UB, LB, clique_nodes, extra_cliques, best_coloring, zLP0 = _bootstrap_initial_from_lp(
#         G, headroom=3, max_k_increase=None, restarts=16, time_limit_bootstrap=min(10.0, time_limit_sec * 0.2), verbose=verbose
#     )
#     allowed_colors: List[int] = list(range(UB))  # Keep K aligned with UB 
#     reserved_colors: Set[int] = set(range(LB))

#     if verbose:
#         print(f"[Init] (from LP-rounding) UB0 = {UB}, LB = {LB}, clique = {clique_nodes}, z_LP0 = {zLP0:.6f}")

#     # maain loop
#     logs: List[Dict[str, Any]] = []
#     no_change_rounds = 0
#     stop_reason = ""

#     for it in range(1, max_rounds + 1):
#         if time.time() - t_all0 > time_limit_sec:
#             stop_reason = "time_limit"
#             break

#         improved = False

#         # step 1: build and solve LP (with precedence + clique cuts)
#         model, var_maps = build_lp_model(
#             G=G, allowed_colors=allowed_colors, clique_nodes=clique_nodes,
#             extra_cliques=extra_cliques, add_precedence=True
#         )
#         info = solve_lp_and_extract(model, var_maps)
#         z_LP, x_frac, y_frac = info["z_LP"], info["x_frac"], info["y_frac"]
#         rc_y = info["rc_y"]
#         if verbose:
#             print(f"[Iter {it}] z_LP={z_LP:.6f} |K|={len(allowed_colors)}")

#         #step 2: multi-start rounding -> verify
#         rounding_coloring = round_and_repair_multi(G, x_frac, y_frac, current_UB=len(allowed_colors), restarts=16, seed=it)
#         rep = verify_coloring(G, rounding_coloring, allowed_colors=list(range(len(allowed_colors))))
#         if verbose:
#             print(f"  [Iter {it} / Rounding] feasible={rep['feasible']}|used_colors={len(set(rounding_coloring.values()))}|conflicts={rep['num_conflicts']}")
#             if not rep["feasible"]:
#                 print(f"  [Iter {it} / Rounding] conflicts_sample ={rep['conflicts_sample'][:10]}")

#         if rep["feasible"]:
#             best_coloring = rounding_coloring
#             used_colors_round = len(set(best_coloring.values()))
#             if used_colors_round < UB:
#                 UB = used_colors_round
#                 allowed_colors = list(range(UB))
#                 improved = True
#                 if verbose:
#                     print(f"  [Round+Repair] Improved UB -> {UB}")
#         # if not feasible, keep previous best_coloring as a safety net.

#         #step 3: conservative fixing (keep K = 0..UB-1)
#         new_allowed = choose_colors_after_fixing(
#             allowed_colors=allowed_colors, rc_y=rc_y, z_LP=z_LP, UB=UB, reserved_colors=reserved_colors
#         )
#         if set(new_allowed) != set(allowed_colors):
#             allowed_colors = new_allowed
#             improved = True
#             if verbose:
#                 print(f"  [Fixing] K changed to {allowed_colors}")

#         #local search: consolidate highest color classes
#         best_coloring, reduced_flag = consolidate_colors(G, best_coloring, passes=5)
#         if reduced_flag:
#             newUB = len(set(best_coloring.values()))
#             if newUB < UB:
#                 UB = newUB
#                 allowed_colors = list(range(UB))
#                 improved = True
#                 if verbose:
#                     print(f"  [LocalSearch] Reduced to UB={UB}")

#         #pure graph-domain UB-1 attempt
#         if not improved:
#             cand, ok = try_ub_minus_one_greedy(G, best_coloring)
#             if ok:
#                 best_coloring = cand
#                 UB = len(set(best_coloring.values()))
#                 allowed_colors = list(range(UB))
#                 improved = True
#                 if verbose:
#                     print(f"  [UB-1 Greedy] success -> UB={UB}")

#         #LP(UB-1) attempt
#         if not improved:
#             newUB, newcol, ok = try_shrink_UB_by_one_LP(G, UB, clique_nodes, extra_cliques, seed=it, verbose=verbose)
#             if ok:
#                 UB = newUB
#                 allowed_colors = list(range(UB))
#                 best_coloring = newcol
#                 improved = True

#         # Logs and invariants
#         logs.append(dict(it=it, UB=UB, LB=LB, z_LP=z_LP, K=len(allowed_colors)))

#         assert UB >= LB, "Invariant broken: UB < LB"
#         assert set(allowed_colors) == set(range(len(allowed_colors))), "Invariant broken: K must be contiguous 0..K-1"
#         assert z_LP <= UB + 1e-6, "LP lower bound exceeds UB"
#         assert verify_coloring(G, best_coloring, allowed_colors=list(range(UB)))["feasible"], "Best coloring infeasible"

#         #stopping criteria
#         if UB <= LB:
#             stop_reason = "UB==LB"
#             if verbose:
#                 print("  [Stop] UB==LB")
#             break

#         no_change_rounds = 0 if improved else (no_change_rounds + 1)
#         if no_change_rounds >= stall_rounds and it >= min_rounds:
#             stop_reason = f"no_progress {no_change_rounds} rounds"
#             if verbose:
#                 print(f"  [Stop] {stop_reason}")
#             break

#         if time.time() - t_all0 > time_limit_sec:
#             stop_reason = "time_limit"
#             break

#     if not stop_reason:
#         stop_reason = "max_rounds"

#     final_report = verify_coloring(G, best_coloring, allowed_colors=list(range(UB)))
#     if verbose:
#         print(f"  [Final] feasible={final_report['feasible']}|used_colors={UB}|conflicts={final_report['num_conflicts']}")

#     return dict(
#         UB=UB, LB=LB, coloring=best_coloring, iters=len(logs), log=logs,
#         stop_reason=stop_reason, feasible=final_report["feasible"], final_check=final_report
#     )