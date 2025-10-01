# graph/local_search.py
from typing import Tuple, Dict, List
from graph.kempe import kempe_chain_component, kempe_swap

def _try_recolor_vertex(G, coloring: Dict[int, int], v: int, target_colors: List[int]) -> bool:
    """Try to recolor v into one of target_colors; greedy first, then Kempe swaps."""
    neigh_cols = {coloring.get(u) for u in G.neighbors(v)}

    #greedy recolor
    for c in target_colors:
        if c not in neigh_cols:
            coloring[v] = c
            return True

    # Kempe swap on two colors
    c_now = coloring[v]
    for c in target_colors:
        if c == c_now:
            continue
        comp = kempe_chain_component(G, coloring, v, c_now, c)
        kempe_swap(coloring, comp, c_now, c)
        ok = all(coloring[w] != coloring[v] for w in G.neighbors(v))
        if ok:
            return True
        #revert if failed
        kempe_swap(coloring, comp, c_now, c)
    return False


def consolidate_colors(G, coloring: Dict[int, int], passes: int = 10) -> Tuple[Dict[int, int], bool]:
    """
    Color consolidation: from high colors downwards, try to pack vertices into lower colors.
    If the highest color class becomes empty, reindex colors to keep 0..k-1 contiguous.
    Returns (new_coloring, reduced?).
    """
    curr = dict(coloring)
    any_reduced = False

    for _ in range(passes):
        colors = sorted(set(curr.values()))
        if len(colors) <= 1:
            break

        #try to clear the highest color class
        maxc = max(colors)
        targets = list(range(maxc))
        vertices = [v for v, c in curr.items() if c == maxc]

        moved_all = True
        for v in vertices:
            if not _try_recolor_vertex(G, curr, v, targets):
                moved_all = False

        if moved_all:
            #remove the emptied color class by reindexing
            used = sorted(set(curr.values()))
            remap = {c: i for i, c in enumerate(used)}
            for v in curr:
                curr[v] = remap[curr[v]]
            any_reduced = True
            print(f"[LocalSearch] highest color class removed; colors now={len(used)}")
        else:
            # try other color classes to facilitate the next pass
            for c in range(maxc - 1, 0, -1):
                targets2 = list(range(c))
                layer = [v for v, col in curr.items() if col == c]
                for v in layer:
                    _try_recolor_vertex(G, curr, v, targets2)

    return curr, any_reduced


def try_reduce_one_color(G, coloring: Dict[int, int]) -> Tuple[Dict[int, int], bool]:
    """compatibility wrapper: attempt consolidation for 2 passes."""
    before = len(set(coloring.values()))
    after_coloring, _ = consolidate_colors(G, coloring, passes=2)
    after = len(set(after_coloring.values()))
    return after_coloring, (after < before)


def lp_guided_pack_highest_color(
    G,
    coloring: Dict[int, int],
    x_frac: Dict[tuple, float],
    K: int,
    passes: int = 2,
    verbose: bool = False, 
) -> Tuple[Dict[int, int], bool]:
    """
    try to empty the highest color class K by moving its vertices into colors 0..K-1,
    guided by LP scores x_frac[(v,c)] (descending). Uses the existing _try_recolor_vertex
    (greedy first, then Kempe).

    NOTE:
    - If not fully emptied, return the ORIGINAL coloring and shrunk=False, so upstream
      logic can safely ignore partial moves.
    """

    curr = dict(coloring)

    # collect vertices at the highest color K BEFORE printing orlogging
    vertices: List[int] = [v for v, c in curr.items() if c == K]
    if verbose:
        print(f"[LP-pack] try packing highest color K={K}, vertices={len(vertices)}")

    if not vertices:
        return coloring, False

    def targets_order_for(v: int) -> List[int]:
        # target colors 0..K-1 sorted by LP preference (x_frac descending)
        cand = list(range(K))
        cand.sort(key=lambda c: x_frac.get((v, c), 0.0), reverse=True)
        return cand

    for _ in range(passes):
        moved_any = False

        # Iterate over a snapshot since we recolor in place
        for v in list(vertices):
            if curr.get(v) != K:
                continue  
            targets = targets_order_for(v)
            if _try_recolor_vertex(G, curr, v, targets):
                moved_any = True

        # recompute remaining vertices at color K
        vertices = [v for v, c in curr.items() if c == K]
        if not vertices:
            # fully emptied -> compact color ids to keep 0..k'-1 contiguous
            used = sorted(set(curr.values()))
            remap = {c: i for i, c in enumerate(used)}
            for u in curr:
                curr[u] = remap[curr[u]]
            if verbose:
                print("[LP-pack] highest color emptied and colors compacted")
            return curr, True

        if not moved_any:
            if verbose:
                print("[LP-pack] no moves possible in this pass")
            break

    return coloring, False
