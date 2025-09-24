from typing import List, Dict, Tuple, Any, Optional
from ortools.linear_solver import pywraplp

def build_lp_model(
    G,
    allowed_colors: List[int],
    clique_nodes: List[int],
    extra_cliques: Optional[List[List[int]]] = None,
    add_precedence: bool = True,
) -> Tuple[pywraplp.Solver, Dict[str, Any]]:
    """
    Assignment-LP:
      min  sum_c y_c
      s.t. sum_c x_{v,c} = 1                        ∀v
           x_{u,c} + x_{v,c} ≤ 1                     ∀(u,v)∈E, ∀c
           x_{v,c} ≤ y_c                             ∀v,c
      + symmetry breaking by clique: v_i uses color i (i=0..|clique|-1)
      + optional precedence: y_{c+1} ≤ y_c
      + optional clique cuts: ∑_{v∈Q} x_{v,c} ≤ 1     ∀Q in extra_cliques, ∀c
    """
    C = list(allowed_colors)
    V = list(G.nodes())
    solver = pywraplp.Solver.CreateSolver("GLOP")
    assert solver is not None

    # variables
    x_vars: Dict[Tuple[int, int], pywraplp.Variable] = {}
    y_vars: Dict[int, pywraplp.Variable] = {}

    for c in C:
        y_vars[c] = solver.NumVar(0.0, 1.0, f"y_{c}")
    for v in V:
        for c in C:
            x_vars[(v, c)] = solver.NumVar(0.0, 1.0, f"x_{v}_{c}")

    # objective
    solver.Minimize(solver.Sum([y_vars[c] for c in C]))

    #1)each vertex exactly one color
    for v in V:
        solver.Add(solver.Sum(x_vars[(v, c)] for c in C) == 1.0)

    # 2) edge constraints per color
    for (u, v) in G.edges():
        for c in C:
            solver.Add(x_vars[(u, c)] + x_vars[(v, c)] <= 1.0)

    #3) linking x <= y
    for v in V:
        for c in C:
            solver.Add(x_vars[(v, c)] <= y_vars[c])

    #4) clique-based symmetry breaking: fix clique_nodes[i] to color i
    t = min(len(clique_nodes), len(C))
    for i in range(t):
        v = clique_nodes[i]
        # enforce x[v,i]=1; x[v,c!=i]=0; y_i=1
        for c in C:
            if c == i:
                solver.Add(x_vars[(v, c)] == 1.0)
                solver.Add(y_vars[c] == 1.0)
            else:
                solver.Add(x_vars[(v, c)] == 0.0)

    # 5) precedence y_{c+1} <= y_c
    if add_precedence:
        for i in range(len(C) - 1):
            solver.Add(y_vars[C[i + 1]] <= y_vars[C[i]])

    #6) clique cuts: ∑_{v∈Q} x_{v,c} <= 1
    if extra_cliques:
        for Q in extra_cliques:
            Qset = set(Q)
            inter = [v for v in V if v in Qset]
            if len(inter) <= 1:
                continue
            for c in C:
                solver.Add(solver.Sum(x_vars[(v, c)] for v in inter) <= 1.0)

    var_maps = dict(x_vars=x_vars, y_vars=y_vars, V=V, C=C)
    return solver, var_maps
