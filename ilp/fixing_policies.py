# ilp/fixing_policies.py
from typing import Dict, Any, List, Tuple, Optional, Iterable

FixPlan = Dict[str, Any]
# structure: {"y_zero": List[int], "x_one": List[Tuple[int,int]], "x_zero": List[Tuple[int,int]]}

def pick_fixings(
    G,
    x_frac: Dict[Tuple[int,int], float],
    y_frac: Dict[int, float],
    z_LP: float,
    UB: int,
    LB: int,
    verbose: bool = True,
    policy: str = "prefix_shrink+strong_assign",
    strong_margin: float = 0.25,
    max_fix_per_round: int = 50,
    rounding_coloring: Optional[Dict[int,int]] = None,
) -> FixPlan:
    parts = [p.strip() for p in policy.split("+") if p.strip()]
    plan: FixPlan = {"y_zero": [], "x_one": [], "x_zero": []}

    # A) prefix_shrink: use ceil(z_LP) to tighten available color prefix
    if "prefix_shrink" in parts:
        K_new = max(LB, int((z_LP + 1e-12)))  # ceil(z_LP)
        if K_new < UB:
            plan["y_zero"].extend(list(range(K_new, UB)))

    # Precompute for each vertex: (best_c, best_val, second_val)
    best_triplets: Dict[int, Tuple[int, float, float]] = {}
    for v in G.nodes():
        vals = [(c, x_frac.get((v,c), 0.0)) for c in range(UB)]
        vals.sort(key=lambda t: t[1], reverse=True)
        if not vals:
            continue
        c1, v1 = vals[0]
        v2 = vals[1][1] if len(vals) >= 2 else 0.0
        best_triplets[v] = (c1, v1, v2)

    # B) strong_assign: strongly confident variable fixing x[v,c]=1
    if "strong_assign" in parts:
        picks = []
        for v, (c1, v1, v2) in best_triplets.items():
            if v1 - v2 >= strong_margin:
                picks.append((v, c1, v1 - v2))
        picks.sort(key=lambda t: t[2], reverse=True)
        for v, c1, _gap in picks[:max_fix_per_round]:
            plan["x_one"].append((v, c1))
        if verbose and plan["x_one"]:
        # estimate how aggressive the picks are by re-reading the current x_frac gaps for the chosen pairs
            gaps = []
            for (v,c) in plan["x_one"][:10]:  # sample first 10 for brevity
                best = x_frac.get((v,c), 0.0)
                second = max(x_frac.get((v,cc), 0.0) for cc in range(UB) if cc != c) if UB > 1 else 0.0
                gaps.append(best - second)
            gmin, gmax = (min(gaps), max(gaps)) if gaps else (0.0, 0.0)
            print(f"  [FixPolicy] sample x-gap min/max over first 10 picks: {gmin:.3f}/{gmax:.3f}")


    # C) rounded_support: if rounding color = best_c in LP and value >=0.8, then fix
    if "rounded_support" in parts and rounding_coloring:
        picks2 = []
        for v, c_round in rounding_coloring.items():
            c1, v1, v2 = best_triplets.get(v, (c_round, 0.0, 0.0))
            if c_round == c1 and v1 >= 0.8:
                picks2.append((v, c_round, v1))
        picks2.sort(key=lambda t: t[2], reverse=True)
        # Avoid over-fitting
        quota = max(1, max_fix_per_round // 2)
        for v, c, _ in picks2[:quota]:
            plan["x_one"].append((v, c))
        if verbose and plan["x_one"]:
            # estimate how aggressive the picks are by re-reading the current x_frac gaps for the chosen pairs
            gaps = []
            for (v,c) in plan["x_one"][:10]:  # sample first 10 for brevity
                best = x_frac.get((v,c), 0.0)
                second = max(x_frac.get((v,cc), 0.0) for cc in range(UB) if cc != c) if UB > 1 else 0.0
                gaps.append(best - second)
            gmin, gmax = (min(gaps), max(gaps)) if gaps else (0.0, 0.0)
            print(f"  [FixPolicy] sample x-gap min/max over first 10 picks: {gmin:.3f}/{gmax:.3f}")


    return plan