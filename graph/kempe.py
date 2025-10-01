# graph/kempe.py

from collections import deque

def kempe_chain_component(G, coloring, u, c1, c2):
    """
    return the component of u in the subgraph induced by colors {c1, c2}:
    vertices reachable from u following only vertices colored c1 or c2.
    """
    comp = set()
    dq = deque([u])
    while dq:
        x = dq.popleft()
        if x in comp:
            continue
        if coloring.get(x) not in (c1, c2):
            continue
        comp.add(x)
        for w in G.neighbors(x):
            if coloring.get(w) in (c1, c2):
                dq.append(w)
    return comp

def kempe_swap(coloring, component, c1, c2):
    for v in component:
        coloring[v] = c2 if coloring[v] == c1 else c1
