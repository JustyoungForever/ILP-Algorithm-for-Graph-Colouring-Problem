from typing import Dict, Tuple, List
from graph.kempe import kempe_chain_component, kempe_swap

def _try_recolor_vertex(G, coloring: Dict[int, int], v: int, targets: List[int]) -> bool:
    neigh_cols = {coloring.get(u) for u in G.neighbors(v)}
    # greedy recolor
    for c in targets:
        if c not in neigh_cols:
            coloring[v] = c
            return True
    # Kempe swap
    c_now = coloring[v]
    for c in targets:
        if c == c_now:
            continue
        comp = kempe_chain_component(G, coloring, v, c_now, c)
        kempe_swap(coloring, comp, c_now, c)
        ok = all(coloring[w] != coloring[v] for w in G.neighbors(v))
        if ok:
            return True
        kempe_swap(coloring, comp, c_now, c)  # revert
    return False

def try_ub_minus_one_greedy(G, coloring: Dict[int, int]) -> Tuple[Dict[int, int], bool]:
    """
    Pure graph-domain attempt to reduce UB by 1:
    Try to pack the highest color class into 0..UB-2.
    If successful, return the reindexed coloring and True; otherwise, return the original coloring and False.
    """
    curr = dict(coloring)
    UB = len(set(curr.values()))
    if UB <= 1:
        return curr, False

    maxc = max(set(curr.values()))
    targets = list(range(maxc))
    vertices = [v for v, c in curr.items() if c == maxc]

    moved_all = True
    for v in vertices:
        if not _try_recolor_vertex(G, curr, v, targets):
            moved_all = False

    if moved_all:
        # success: reindex colors to keep 0..k-1 contiguous
        used = sorted(set(curr.values()))
        remap = {c: i for i, c in enumerate(used)}
        for v in curr:
            curr[v] = remap[curr[v]]
        return curr, True
    return coloring, False
