import networkx as nx
from typing import Dict

def dsatur_coloring(G) -> Dict[int, int]:
    """
    NetworkX greedy coloring with strategy='DSATUR'.
    Returns dict node->color_id (0..k-1); compacts color ids to be contiguous.
    """
    raw = nx.coloring.greedy_color(G, strategy="DSATUR")
    used = sorted(set(raw.values()))
    remap = {c: i for i, c in enumerate(used)}
    return {v: remap[c] for v, c in raw.items()}
