# ilp/incremental.py
from typing import Any, Dict, List, Tuple, Iterable, Optional
from ortools.linear_solver import pywraplp
from ilp.model import build_lp_model
from ilp.lp_solve import solve_lp_and_extract

BoundsToken = List[Tuple[pywraplp.Variable, float, float]]

class IncrementalLP:
    def __init__(
        self, G, allowed_colors: List[int], clique_nodes: List[int],
        extra_cliques: Optional[List[List[int]]] = None, add_precedence: bool = True,
        edge_mode: str = "auto", lazy_threshold: float = 2e7,
    ):
        self.G = G
        self.solver, self.var_maps = build_lp_model(
            G=G, allowed_colors=allowed_colors, clique_nodes=clique_nodes,
            extra_cliques=extra_cliques, add_precedence=add_precedence,
            edge_mode=edge_mode, lazy_threshold=lazy_threshold,
        )
        self.x_vars = self.var_maps["x_vars"]
        self.y_vars = self.var_maps["y_vars"]
        self.V = self.var_maps["V"]
        self.C = self.var_maps["C"]

        self.lazy_edges = bool(self.var_maps.get("lazy_edges", False))
        self._edge_cut_added = set()  # (min(u,v), max(u,v), c)
    def set_time_limit_ms(self, ms: int) -> None:
        try:
            self.solver.SetTimeLimit(int(max(0, ms)))
        except Exception:
            pass

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
    
    def add_edge_color_cut(self, u: int, v: int, c: int) -> bool:
        """Add constraint x[u,c] + x[v,c] <= y[c] once. Return True if new."""
        if u == v:
            return False
        a, b = (u, v) if u < v else (v, u)
        key = (a, b, c)
        if key in self._edge_cut_added:
            return False
        if (u, c) not in self.x_vars or (v, c) not in self.x_vars:
            return False
        if c not in self.y_vars:
            return False

        self.solver.Add(self.x_vars[(u, c)] + self.x_vars[(v, c)] <= self.y_vars[c])
        self._edge_cut_added.add(key)
        return True

    def lazy_separate_from_lp(self, x_frac: dict, y_frac: dict,
                              top_t: int = 2, eps: float = 1e-6, max_new: int = 200000) -> int:
        """
        Cut generation:
          for each edge (u,v), only check colors in intersection of top_t colors of u and v;
          if x[u,c] + x[v,c] > y[c] + eps => add cut.
        """
        if not self.lazy_edges:
            return 0

        # pick top_t colors per vertex
        top = {}
        for v in self.V:
            best = []  # list of (val, c)
            for c in self.C:
                val = x_frac.get((v, c), 0.0)
                if len(best) < top_t:
                    best.append((val, c))
                    if len(best) == top_t:
                        best.sort()
                else:
                    if val > best[0][0]:
                        best[0] = (val, c)
                        best.sort()
            top[v] = {c for (_, c) in best}

        added = 0
        for (u, v) in self.G.edges():
            iu = top.get(u)
            iv = top.get(v)
            if not iu or not iv:
                continue
            for c in (iu & iv):
                if x_frac.get((u, c), 0.0) + x_frac.get((v, c), 0.0) > y_frac.get(c, 1.0) + eps:
                    if self.add_edge_color_cut(u, v, c):
                        added += 1
                        if added >= max_new:
                            return added
        return added

    def lazy_add_conflict_cuts(self, coloring: dict, max_new: int = 50000) -> int:
        """If a candidate coloring has conflicts, add those (u,v,c) cuts."""
        if not self.lazy_edges:
            return 0
        added = 0
        for (u, v) in self.G.edges():
            cu = coloring.get(u, None)
            cv = coloring.get(v, None)
            if cu is None or cv is None:
                continue
            if cu == cv and cu in self.y_vars:
                if self.add_edge_color_cut(u, v, int(cu)):
                    added += 1
                    if added >= max_new:
                        break
        return added

    def solve(self) -> Dict[str, Any]:
        return solve_lp_and_extract(self.solver, self.var_maps)

    def try_apply_and_solve(self, tokens: List[BoundsToken]) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Apply a batch of modifications and try to solve; if not optimal (e.g., infeasible), revert and return (None, False)."""
        try:
            info = solve_lp_and_extract(self.solver, self.var_maps)  # 内部 solve + extract
            return info, True
        except TimeoutError:
            return None, False

    def revert_all(self, tokens: List[BoundsToken]) -> None:
        for tok in reversed(tokens):
            self._revert(tok)