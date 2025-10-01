# graph/slo.py
import networkx as nx
from typing import Dict

def smallest_last_coloring(G) -> Dict[int, int]:
    raw = nx.coloring.greedy_color(G, strategy="smallest_last")
    used = sorted(set(raw.values()))
    remap = {c: i for i, c in enumerate(used)}
    return {v: remap[c] for v, c in raw.items()}