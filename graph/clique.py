import networkx as nx
from typing import List

def greedy_max_clique(G) -> List[int]:
    """
    Simple greedy heuristic for a maximal clique (not guaranteed to be maximum).
    """
    # order nodes by degree (desc)
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    clique = []
    for v in nodes:
        if all(G.has_edge(v, u) for u in clique):
            clique.append(v)
    return clique
