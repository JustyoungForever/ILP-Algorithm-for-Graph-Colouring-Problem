# ilp/lp_solve.py
from typing import Dict, Any, Tuple
from ortools.linear_solver import pywraplp

def solve_lp_and_extract(solver: pywraplp.Solver, var_maps: Dict[str, Any]) -> Dict[str, Any]:
    """
    solve LP with GLOP and extract:
      - z_LP
      - x_frac: dict[(v,c)] -> float
      - y_frac:dict[c] -> float
      - rc_y: dict[c] -> reduced cost of y_c
    """
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        pass
    elif status == pywraplp.Solver.NOT_SOLVED:
        # Most commonly: time limit reached or solver stopped early
        raise TimeoutError(f"LP not solved (status={status})")
    else:
        # INFEASIBLE / UNBOUNDED / ABNORMAL / MODEL_INVALID
        raise RuntimeError(f"LP failed (status={status})")

    x_vars = var_maps["x_vars"]
    y_vars = var_maps["y_vars"]
    V = var_maps["V"]
    C = var_maps["C"]

    # extract
    x_frac = {(v, c): x_vars[(v, c)].solution_value() for v in V for c in C}
    y_frac = {c: y_vars[c].solution_value() for c in C}
    z_LP = solver.Objective().Value()
    return {"z_LP": z_LP, "x_frac": x_frac, "y_frac": y_frac, "status": status}