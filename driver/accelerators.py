# driver/accelerators.py
from typing import Dict, Tuple
from graph.verify import verify_coloring
from graph.local_search import consolidate_colors, lp_guided_pack_highest_color
from heuristics.round_and_repair import round_and_repair_multi
from graph.ub1_greedy import try_ub_minus_one_greedy
# driver/accelerators.py
from visualisierung.draw import visualize_coloring

def _debug_candidate_from_xfrac(x_frac, K_eff: int, tau: float = 0.10):
    """
    兼容多种 x_frac 结构，生成“倾向着色”：
      - 结构A: {(v,c): float}
      - 结构B: {v: {c: float}}
      - 结构C: {v: list/array}  (下标=颜色)
    仅使用颜色域 [0..K_eff-1]；最大分量 >= tau 才上色；其余节点不着色。
    """
    cand = {}
    if not x_frac:
        return cand

    # 尝试判断结构A: key 是 (v,c) 元组
    try:
        sample_key = next(iter(x_frac.keys()))
    except StopIteration:
        return cand

    if isinstance(sample_key, tuple) and len(sample_key) == 2:
        # A: {(v,c): val}
        best = {}  # v -> (best_c, best_val)
        for (v, c), val in x_frac.items():
            try:
                c_int = int(c)
                val_f = float(val)
            except Exception:
                continue
            if c_int < 0 or c_int >= K_eff:
                continue
            if (v not in best) or (val_f > best[v][1]):
                best[v] = (c_int, val_f)
        for v, (c_int, val_f) in best.items():
            if val_f >= tau:
                cand[v] = c_int
        return cand

    # 结构B/C: {v: dict/list/array}
    for v, row in x_frac.items():
        items = []
        if isinstance(row, dict):
            for c, val in row.items():
                try:
                    c_int = int(c)
                    val_f = float(val)
                except Exception:
                    continue
                if 0 <= c_int < K_eff:
                    items.append((c_int, val_f))
        elif hasattr(row, "__len__") and hasattr(row, "__getitem__"):
            # list/tuple/ndarray 等
            try:
                L = len(row)
            except Exception:
                L = 0
            for c_int in range(min(L, K_eff)):
                try:
                    val_f = float(row[c_int])
                except Exception:
                    continue
                items.append((c_int, val_f))
        else:
            continue

        if not items:
            continue
        c_max, val_max = max(items, key=lambda kv: kv[1])
        if val_max >= tau:
            cand[v] = int(c_max)

    return cand



def _viz_internal(G, step: str, round_id: int, cmap, allowed_colors: int,
                  out_dir="visualisierung", layout_seed=42,*, only_colored: bool = True):
    """
    统一内部可视化：只要 cmap 非空就出图（仅可视化已上色子图；冲突/越界点=黑色）。
    allowed_colors 传整数 K → 自动使用 [0..K-1]。
    """
    if not cmap:
        return
    visualize_coloring(
        G, cmap, step=step, round_id=round_id,
        out_dir=out_dir, layout_seed=layout_seed,
        only_colored=only_colored,  
        allowed_colors=list(range(max(1, allowed_colors))),
        conflict_nodes_fill_black=True,
    )

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
    restarts: int, perturb_y: float, seed: int, verbose: bool = False,
    enable_visualization: bool = False, viz_round_id: int = 0,
    viz_out_dir: str = "visualisierung", viz_layout_seed: int = 42,
    step_name: str = "Side-K+1"
) -> Tuple[int, Dict[int,int], bool]:
    """
    temporarily use K+1 colors for rounding while keeping LP locked at K;
    returns improvement if a smaller UB is constructed.
    """
    side_colors = K + 1
    if verbose and side_colors > UB:
        print(f"  [Side] skip: K+1={side_colors} exceeds current UB={UB}")
    if side_colors > UB:
        if enable_visualization:
            cand_dbg = _debug_candidate_from_xfrac(x_frac, K_eff=side_colors, tau=0.10)
            _viz_internal(G, f"{step_name}-failed", viz_round_id, cand_dbg, allowed_colors=side_colors,
                        out_dir=viz_out_dir, layout_seed=viz_layout_seed, only_colored=False)
        return UB, {}, False

    cand = round_and_repair_multi(
        G, x_frac, y_frac, current_UB=side_colors,
        restarts=max(4, restarts // 2), seed=seed, perturb_y=perturb_y
    )
    rep = verify_coloring(G, cand, allowed_colors=list(range(side_colors)))
    if verbose and not rep["feasible"]:
        print("  [Side] K+1 rounding produced an infeasible coloring (will not update UB)")
        if enable_visualization:
            to_draw = cand if cand else _debug_candidate_from_xfrac(x_frac, K_eff=side_colors, tau=0.10)
            _viz_internal(G, f"{step_name}-failed", viz_round_id, to_draw, allowed_colors=side_colors,
                          out_dir=viz_out_dir, layout_seed=viz_layout_seed)
        return UB, {}, False

    # Compact + small local search
    cand, used = compact_colors(cand)
    cand2, reduced = consolidate_colors(G, dict(cand), passes=3)
    if reduced:
        cand2, used = compact_colors(cand2)
        cand = cand2

    # 可视化（成功或可行但不更好）
    if enable_visualization:
        _viz_internal(G, step_name, viz_round_id, cand, allowed_colors=side_colors,
                      out_dir=viz_out_dir, layout_seed=viz_layout_seed)

    if used < UB:
        if verbose:
            print(f"  [UB-1 side attempt] improved UB -> {used} (using {side_colors} colors)")
        return used, cand, True
    if verbose and used >= UB:
        print(f"  [Side] feasible but not better (used={used} ≥ UB={UB})")
    return UB, {}, False



def try_lp_guided_pack(
    G, best_coloring: Dict[int,int], x_frac, K: int, UB: int, *, verbose: bool = False,
    enable_visualization: bool = False, viz_round_id: int = 0,
    viz_out_dir: str = "visualisierung", viz_layout_seed: int = 42
) -> Tuple[int, Dict[int,int], bool]:
    """
    only called when UB == K+1: use LP's x_frac to guide packing of highest color layer (K) back into 0..K-1.
    success means UB becomes <= K.
    """
    if verbose:
        print(f"  [LP-guided pack] trying to empty highest color K={K} (only active if UB==K+1)")

    if UB != K + 1:
        # --- 可视化：条件不满足也给一张失败倾向图（基于 x_frac） ---
        if enable_visualization:
            cand_dbg = _debug_candidate_from_xfrac(x_frac, K_eff=K, tau=0.10)
            _viz_internal(G, "LP-Guided-Pack-failed", viz_round_id, cand_dbg, allowed_colors=K,
                        out_dir=viz_out_dir, layout_seed=viz_layout_seed, only_colored=False)

        return UB, {}, False

    packed, shr = lp_guided_pack_highest_color(G, best_coloring, x_frac, K, passes=2)
    if verbose and not shr:
        print("  [LP-guided pack] no full emptying of highest color — no UB reduction")
        # --- 可视化：失败也出图（用 x_frac 还原） ---
        if enable_visualization:
            cand_dbg = _debug_candidate_from_xfrac(x_frac, K_eff=K, tau=0.10)
            _viz_internal(G, "LP-Guided-Pack-failed", viz_round_id, cand_dbg, allowed_colors=K,
                          out_dir=viz_out_dir, layout_seed=viz_layout_seed)
        return UB, {}, False

    packed, used = compact_colors(packed)

    # --- 可视化：成功（pack 后的解） ---
    if enable_visualization:
        _viz_internal(G, "LP-Guided-Pack", viz_round_id, packed, allowed_colors=K,
                      out_dir=viz_out_dir, layout_seed=viz_layout_seed)

    if used < UB:
        if verbose:
            print(f"  [LP-guided pack] UB -> {used}")
        return used, packed, True
    return UB, {}, False


def try_graph_ub1_greedy(
    G, best_coloring: Dict[int,int], UB: int, *, verbose: bool = False,
    enable_visualization: bool = False, viz_round_id: int = 0,
    viz_out_dir: str = "visualisierung", viz_layout_seed: int = 42
) -> Tuple[int, Dict[int,int], bool]:
    """
    pure graph-domain UB-1 greedy (complementary to LP-guided). Success reduces UB by 1.
    """
    if verbose:
        print("  [UB-1 greedy] attempting to fold top color class graphically")
    cand, ok = try_ub_minus_one_greedy(G, best_coloring)
    if not ok:
        # --- 可视化：没有候选就用 best 做 -nc 占位图（此函数拿不到 x_frac） ---
        if enable_visualization:
            _viz_internal(G, "UB-1-Greedy-nc", viz_round_id, best_coloring, allowed_colors=max(1, UB-1),
                          out_dir=viz_out_dir, layout_seed=viz_layout_seed)
        return UB, {}, False

    cand, used = compact_colors(cand)

    # --- 可视化：成功或可行但不更好，都画 cand ---
    if enable_visualization:
        _viz_internal(G, "UB-1-Greedy", viz_round_id, cand, allowed_colors=max(1, UB-1),
                      out_dir=viz_out_dir, layout_seed=viz_layout_seed)

    if used < UB:
        if verbose:
            print(f"  [UB-1 greedy] UB -> {used}")
        return used, cand, True
    if verbose and used >= UB:
        print(f"  [UB-1 greedy] feasible but not better (used={used} >= UB={UB})")
    return UB, {}, False
