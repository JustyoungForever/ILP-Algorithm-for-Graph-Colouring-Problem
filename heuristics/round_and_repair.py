# heuristics/round_and_repair.py
from typing import Dict, List
import random
import math
from graph.kempe import kempe_chain_component, kempe_swap
import time
from typing import Optional


def round_and_repair(G, x_frac: Dict[tuple, float], y_frac: Dict[int, float], current_UB: int,    *,
    tie_seed: int = 0,vertex_rcl: int = 1,color_rcl: int = 1,x_jitter: float = 0.0,color_tau: float = 0.0,) -> Dict[int, int]:
    """
    LP-guided rounding with DSATUR ordering:
      1) Assign colors vertex-by-vertex in DSATUR order; per-vertex color choice prioritizes x_frac, with y_frac as a global tie-breaker.
      2) If no color is currently available for a vertex, attempt a single Kempe swap; if still impossible, place the argmax color and defer to the repair phase.
      3) Finally, run a small conflict-repair loop (expected to be shorter than naive rounding).
    """
    V = list(G.nodes())
    colors = list(range(current_UB))
    # global color priority (higher y first)
    ranked_colors = sorted(colors, key=lambda c: y_frac.get(c, 0.0), reverse=True)

    #Ordered rounding (DSATUR)
    # Track: uncolored set, saturation sets, degrees
    uncolored = set(V)
    colored: Dict[int, int] = {}
    sat_sets = {v: set() for v in V}  # for v: set of colors used by its colored neighbors
    deg = {v: G.degree(v) for v in V}

    rnd = random.Random(tie_seed)
    def softmax_choice(scored, tau: float, topk: int):
        """
        scored: List[(score, item)]  score越大越好
        tau: temperature > 0
        topk: 只在topk里采样（加速 + 稳定）；topk>=1
        """
        if not scored:
            return None
        # 只取前topk
        if topk is not None and topk > 0 and len(scored) > topk:
            scored = sorted(scored, key=lambda t: t[0], reverse=True)[:topk]

        # 数值稳定 softmax
        m = max(s for s, _ in scored)
        weights = [math.exp((s - m) / tau) for s, _ in scored]
        total = sum(weights)
        r = rnd.random() * total
        acc = 0.0
        for (_, item), w in zip(scored, weights):
            acc += w
            if acc >= r:
                return item
        return scored[-1][1]

    # Precompute max x value per vertex (independent of y order), to avoid O(|V|*|C|) scan inside pick_next_vertex.
    max_x = {v: 0.0 for v in V}
    for (vv, cc), val in x_frac.items():
        if 0 <= cc < current_UB and vv in max_x:
            if val > max_x[vv]:
                max_x[vv] = val

    # def pick_next_vertex():
    #     # pick the vertex with highest saturation; tie-break: degree -> max x_frac -> tiny jitter
    #     best = None
    #     best_key = None
    #     for v in uncolored:
    #         sat = len(sat_sets[v])
    #         best_c_val = 0.0
    #         for c in ranked_colors:
    #             best_c_val = max(best_c_val, x_frac.get((v, c), 0.0))
    #         key = (sat, deg[v], best_c_val, rnd.random()*1e-9)
    #         if best_key is None or key > best_key:
    #             best_key = key
    #             best = v
    #     return best
    def pick_next_vertex():
        # RCL over vertices: rank by (saturation, degree, max_x[v]) then pick randomly from top vertex_rcl
        scored = []
        for v in uncolored:
            sat = len(sat_sets[v])

            # keep default behavior when x_jitter == 0
            x_score = max_x[v]
            if x_jitter > 0.0:
                # very small stochastic tie-break / diversification guided by x
                x_score = x_score + x_jitter * rnd.random()

            key = (sat, deg[v], x_score, rnd.random() * 1e-12)
            scored.append((key, v))

        scored.sort(reverse=True)
        r = min(max(1, vertex_rcl), len(scored))
        if r == 1:
            return scored[0][1]
        return rnd.choice([v for _, v in scored[:r]])


    while uncolored:
        v = pick_next_vertex()
        neigh_cols = set(colored.get(u) for u in G.neighbors(v) if u in colored)
        # choose the best available color by x_frac among colors not in neigh_cols
        # candidate = None
        # best_val = -1.0
        # for c in ranked_colors:
        #     if c in neigh_cols:
        #         continue
        #     val = x_frac.get((v, c), 0.0)
        #     if val > best_val:
        #         best_val = val
        #         candidate = c
        avail = []
        for c in ranked_colors:
            if c in neigh_cols:
                continue
            val = x_frac.get((v, c), 0.0)
            if x_jitter > 0.0:
                val = val + x_jitter * rnd.random()
            avail.append((val, c))

        candidate = None
        if avail:
            if color_tau > 0.0:
                # softmax sampling over top-k pool (topk=color_rcl)
                candidate = softmax_choice(avail, tau=color_tau, topk=max(1, color_rcl))
            else:
                # old behavior: greedy / RCL-topk-uniform
                avail.sort(reverse=True)
                r = min(max(1, color_rcl), len(avail))
                if r == 1:
                    candidate = avail[0][1]
                else:
                    candidate = rnd.choice([c for _, c in avail[:r]])


        assigned = False
        if candidate is not None:
            colored[v] = candidate
            assigned = True
        else:
            #ifno available color: try a Kempe swap
            # want = max(ranked_colors, key=lambda c: x_frac.get((v, c), 0.0))
            want_scored = []
            for c in ranked_colors:
                val = x_frac.get((v, c), 0.0)
                if x_jitter > 0.0:
                    val = val + x_jitter * rnd.random()
                want_scored.append((val, c))

            if color_tau > 0.0:
                want = softmax_choice(want_scored, tau=color_tau, topk=max(1, color_rcl))
            else:
                want_scored.sort(reverse=True)
                r = min(max(1, color_rcl), len(want_scored))
                want = want_scored[0][1] if r == 1 else rnd.choice([c for _, c in want_scored[:r]])
          
            # attempt a Kempe swap between 'want' and any neighbor color
            tried = False
            for c2 in sorted(neigh_cols, key=lambda c: -x_frac.get((v, c), 0.0)):
                if c2 == want:
                    continue
                comp = kempe_chain_component(G, {**colored, v: want}, v, want, c2)
                tmp = dict(colored); tmp[v] = want
                kempe_swap(tmp, comp, want, c2)
                if all(tmp.get(u) != tmp[v] for u in G.neighbors(v)):
                    colored = tmp
                    assigned = True
                    tried = True
                    break
                # if not tried and not assigned:
                #     # still no luck: place argmax color and defer fix
                #     colored[v] = want
                #     assigned = True
                if not tried and not assigned:
                    # BIG FIX: do NOT force 'want' (creates conflicts).
                    # Choose a color that minimizes immediate conflicts with already-colored neighbors.
                    best_c = None
                    best_key = None
                    for c in ranked_colors:
                        conf = 0
                        for u in G.neighbors(v):
                            if u in colored and colored[u] == c:
                                conf += 1
                        # prefer fewer conflicts; tie-break by x and y
                        key = (-conf, x_frac.get((v, c), 0.0), y_frac.get(c, 0.0), rnd.random() * 1e-12)
                        if best_key is None or key > best_key:
                            best_key = key
                            best_c = c

                    colored[v] = best_c if best_c is not None else want
                    assigned = True


        # update saturation info
        uncolored.remove(v)
        for u in G.neighbors(v):
            if u in uncolored and colored.get(v) is not None:
                sat_sets[u].add(colored[v])

    #just repair a small number of remaining conflicts
    cand = colored

    def conflicts():
        for u, w in G.edges():
            if cand[u] == cand[w]:
                yield (u, w)

    iters, MAX_ITERS = 0, (80 if len(V) >= 500 else max(3 * len(V), 150))

    while True:
        bad = []
        # BIG: cap conflict collection to avoid full edge scan each iter
        cap = 200 if len(V) >= 500 else 10**9
        for e in conflicts():
            bad.append(e)
            if len(bad) >= cap:
                break

        if not bad:
            break
        iters += 1
        if iters > MAX_ITERS:
            break

        # pick the endpoint harder to recolor (fewer options, higher degree)
        u, w = max(bad, key=lambda e: (deg[e[0]] + deg[e[1]]))
        pick = u if deg[u] >= deg[w] else w
        used_neigh = {cand[t] for t in G.neighbors(pick)}
        moved = False
        # try a free color (prioritized by y_frac)
        # for c_try in ranked_colors:
        #     if c_try not in used_neigh:
        #         cand[pick] = c_try
        #         moved = True
        #         break
        if (color_tau <= 0.0) and (color_rcl == 1) and (x_jitter == 0.0):
            # default behavior: y-priority
            for c_try in ranked_colors:
                if c_try not in used_neigh:
                    cand[pick] = c_try
                    moved = True
                    break
        else:
            free = []
            for c_try in ranked_colors:
                if c_try in used_neigh:
                    continue
                val = x_frac.get((pick, c_try), 0.0)
                if x_jitter > 0.0:
                    val = val + x_jitter * rnd.random()
                free.append((val, c_try))

            if free:
                if color_tau > 0.0:
                    cand[pick] = softmax_choice(free, tau=color_tau, topk=max(1, color_rcl))
                else:
                    free.sort(reverse=True)
                    r = min(max(1, color_rcl), len(free))
                    cand[pick] = free[0][1] if r == 1 else rnd.choice([c for _, c in free[:r]])
                moved = True

        if moved:
            continue
        # fallback to Kempe
        c_now = cand[pick]
        for c_try in ranked_colors:
            if c_try == c_now:
                continue
            comp = kempe_chain_component(G, cand, pick, c_now, c_try)
            kempe_swap(cand, comp, c_now, c_try)
            ok = all(cand[t] != cand[pick] for t in G.neighbors(pick))
            if ok:
                moved = True
                break
            kempe_swap(cand, comp, c_now, c_try)  # revert one more time

    return cand


# heuristics/round_and_repair.py 仅改 wrap 多启动函数
def round_and_repair_multi(G, x_frac, y_frac, current_UB, restarts=36, seed=0, perturb_y: float = 1e-6,deadline_ts: Optional[float] = None,*,
                               diversify_ties: bool = False,debug: bool = False,debug_tag: str = "RNR",vertex_rcl: int = 1,
                               color_rcl: int = 1,x_jitter: float = 0.0, color_tau: float = 0.0):
    best = None
    best_used = 10**9
    rnd = random.Random(seed)

    used_list = []
    sigs = set()
    V_order = list(G.nodes())

    for r in range(restarts):
        if deadline_ts is not None and time.monotonic() >= deadline_ts:
            break

        # jitter only affects ranked color order (by y)
        y_jitter = {c: y_frac.get(c, 0.0) + perturb_y * rnd.random() for c in range(current_UB)}

        # NEW: optional DSATUR tie-break diversification
        tie_seed = (seed * 1000003 + r) if diversify_ties else 0
        if debug and r == 0:
            print(f"  [{debug_tag}] cfg vertex_rcl={vertex_rcl} color_rcl={color_rcl} "
                  f"x_jitter={x_jitter:.1e} color_tau={color_tau:.3f} diversify_ties={diversify_ties}")

        cand = round_and_repair(
            G, x_frac, y_jitter, current_UB,
            tie_seed=tie_seed,
            vertex_rcl=vertex_rcl,
            color_rcl=color_rcl,
            x_jitter=x_jitter,
            color_tau=color_tau,
        )
        if deadline_ts is not None and time.monotonic() >= deadline_ts:
            # 防止单次 restart 跑穿后还继续做统计/签名等
            if best is None:
                best = cand  # 至少把已经算出来的解带回去
            break

        used = len(set(cand.values()))
        used_list.append(used)

        # cheap diversity proxy: hash assignment vector
        try:
            sigs.add(hash(tuple(cand.get(v, -1) for v in V_order)))
        except Exception:
            pass

        if used < best_used:
            best_used = used
            best = cand

    if debug and used_list:
        uniq_ratio = (len(sigs) / max(1, len(used_list))) if sigs else 0.0
        mean_used = sum(used_list) / len(used_list)
        print(
            f"  [{debug_tag}] restarts_done={len(used_list)}/{restarts} "
            f"used_min={min(used_list)} used_mean={mean_used:.2f} "
            f"unique_ratio={uniq_ratio:.2f} best_used={best_used}"
        )
    # --- GUARANTEE: never return None ---
    if best is None:
        # 说明一次 restart 都没执行（通常是 deadline 已过）
        # 返回一个极快的兜底 coloring：颜色范围严格在 [0, current_UB-1]
        V = list(G.nodes())
        best = {v: (i % current_UB) for i, v in enumerate(V)}
        if debug:
            print(f"  [{debug_tag}] WARNING: no restart executed; returning fallback coloring used={len(set(best.values()))}")
    return best

