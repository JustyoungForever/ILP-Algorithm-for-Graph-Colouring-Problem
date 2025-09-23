from typing import Dict, List
import random
from graph.kempe import kempe_chain_component, kempe_swap

def round_and_repair(G, x_frac: Dict[tuple, float], y_frac: Dict[int, float], current_UB: int) -> Dict[int, int]:
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

    # ---------- Ordered rounding (DSATUR) ----------
    # Track: uncolored set, saturation sets, degrees
    uncolored = set(V)
    colored: Dict[int, int] = {}
    sat_sets = {v: set() for v in V}  # for v: set of colors used by its colored neighbors
    deg = {v: G.degree(v) for v in V}

    rnd = random.Random(0)

    def pick_next_vertex():
        # pick the vertex with highest saturation; tie-break: degree -> max x_frac -> tiny jitter
        best = None
        best_key = None
        for v in uncolored:
            sat = len(sat_sets[v])
            best_c_val = 0.0
            for c in ranked_colors:
                best_c_val = max(best_c_val, x_frac.get((v, c), 0.0))
            key = (sat, deg[v], best_c_val, rnd.random()*1e-9)
            if best_key is None or key > best_key:
                best_key = key
                best = v
        return best

    while uncolored:
        v = pick_next_vertex()
        neigh_cols = set(colored.get(u) for u in G.neighbors(v) if u in colored)
        # choose the best available color by x_frac among colors not in neigh_cols
        candidate = None
        best_val = -1.0
        for c in ranked_colors:
            if c in neigh_cols:
                continue
            val = x_frac.get((v, c), 0.0)
            if val > best_val:
                best_val = val
                candidate = c

        assigned = False
        if candidate is not None:
            colored[v] = candidate
            assigned = True
        else:
            # no available color: try a Kempe swap
            want = max(ranked_colors, key=lambda c: x_frac.get((v, c), 0.0))
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
            if not tried and not assigned:
                # still no luck: place argmax color and defer fix
                colored[v] = want
                assigned = True

        # update saturation info
        uncolored.remove(v)
        for u in G.neighbors(v):
            if u in uncolored and colored.get(v) is not None:
                sat_sets[u].add(colored[v])

    # ---------- Repair a small number of remaining conflicts ----------
    cand = colored

    def conflicts():
        for u, w in G.edges():
            if cand[u] == cand[w]:
                yield (u, w)

    iters, MAX_ITERS = 0, max(3 * len(V), 150)
    while True:
        bad = list(conflicts())
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
        for c_try in ranked_colors:
            if c_try not in used_neigh:
                cand[pick] = c_try
                moved = True
                break
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
            kempe_swap(cand, comp, c_now, c_try)  # revert

    return cand


def round_and_repair_multi(G, x_frac, y_frac, current_UB, restarts=16, seed=0) -> Dict[int, int]:
    """Multi-start wrapper: jitter y_frac to diversify tie-breaks; keep the best (fewest colors used)."""
    best = None
    best_used = 10**9
    rnd = random.Random(seed)
    for _ in range(restarts):
        y_jitter = {c: y_frac.get(c, 0.0) + 1e-6 * rnd.random() for c in range(current_UB)}
        cand = round_and_repair(G, x_frac, y_jitter, current_UB)
        used = len(set(cand.values()))
        if used < best_used:
            best_used = used
            best = cand
    return best
