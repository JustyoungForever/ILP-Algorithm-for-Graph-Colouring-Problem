# graph/cliques_enum.py
from typing import List
import time
import networkx as nx

def top_maximal_cliques(G, max_cliques: int = 50, min_size: int = 4, time_limit_sec: float = 2.0) -> List[list]:
    """
    Iterate maximal cliques and collect a handful of reasonably large ones.
    Controls:
      - at most `max_cliques` cliques,
      - minimum size `min_size`,
      - overall time budget `time_limit_sec`.
    """
    cliques = []
    t0 = time.time()
    for Q in nx.find_cliques(G):
        if time.time() - t0 > time_limit_sec:
            break
        if len(Q) >= min_size:
            cliques.append(Q)
            if len(cliques) >= max_cliques:
                break
    # deduplicate (by set)
    uniq = []
    seen = set()
    for Q in cliques:
        key = tuple(sorted(Q))
        if key not in seen:
            seen.add(key)
            uniq.append(list(key))
    return uniq
