# ilp/incremental.py
from typing import Any, Dict, List, Tuple, Iterable, Optional
from ortools.linear_solver import pywraplp
from ilp.model import build_lp_model
from ilp.lp_solve import solve_lp_and_extract

BoundsToken = List[Tuple[pywraplp.Variable, float, float]]

class IncrementalLP:
    def __init__(self, G, allowed_colors: List[int], clique_nodes: List[int],
                 extra_cliques: Optional[List[List[int]]] = None, add_precedence: bool = True):
        self.solver, self.var_maps = build_lp_model(
            G=G, allowed_colors=allowed_colors, clique_nodes=clique_nodes,
            extra_cliques=extra_cliques, add_precedence=add_precedence
        )
        self.x_vars = self.var_maps["x_vars"]  # (v,c) -> var
        self.y_vars = self.var_maps["y_vars"]  # c -> var
        self.V = self.var_maps["V"]
        self.C = self.var_maps["C"]

    def _snapshot(self, vars: Iterable[pywraplp.Variable]) -> BoundsToken:
        tok: BoundsToken = []
        for var in vars:
            tok.append((var, var.Lb(), var.Ub()))
        return tok

    def _revert(self, token: BoundsToken) -> None:
        for var, lb, ub in token:
            var.SetBounds(lb, ub)

    def fix_y_zero(self, colors: Iterable[int]) -> BoundsToken:
        token: BoundsToken = []
        for c in colors:
            v = self.y_vars.get(c)
            if v is not None:
                token += self._snapshot([v])
                v.SetBounds(0.0, 0.0)
        return token

    def fix_vertex_color(self, vtx: int, c_star: int) -> BoundsToken:
        # x[v,c*] = 1; x[v,c!=c*]=0 ensure sum_c x[v,c]=1 and linking constraints consistent
        token: BoundsToken = []
        for c in self.C:
            var = self.x_vars[(vtx, c)]
            token += self._snapshot([var])
            if c == c_star:
                var.SetBounds(1.0, 1.0)
            else:
                var.SetBounds(0.0, 0.0)
        return token

    def forbid_x(self, vtx: int, c: int) -> BoundsToken:
        var = self.x_vars[(vtx, c)]
        tok = self._snapshot([var])
        var.SetBounds(0.0, 0.0)
        return tok

    def lock_prefix_K(self, K: int) -> BoundsToken:
        # y_c = 0 for c >= K; others keep current bounds
        colors = [c for c in self.C if c >= K]
        return self.fix_y_zero(colors)

    def solve(self) -> Dict[str, Any]:
        return solve_lp_and_extract(self.solver, self.var_maps)

    def try_apply_and_solve(self, tokens: List[BoundsToken]) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Apply a batch of modifications and try to solve; if not optimal (e.g., infeasible), revert and return (None, False)."""
        status = self.solver.Solve()
        # Solve once first (warm start), then actually solve and extract
        if status != pywraplp.Solver.OPTIMAL:
            # If cannot solve even without modifications, fail directly
            return None, False
        info = solve_lp_and_extract(self.solver, self.var_maps)
        return info, True

    def revert_all(self, tokens: List[BoundsToken]) -> None:
        for tok in reversed(tokens):
            self._revert(tok)