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
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"LP not optimal (status={status})")

    z_LP = solver.Objective().Value()
    x_vars = var_maps["x_vars"]
    y_vars = var_maps["y_vars"]
    C = var_maps["C"]

    x_frac = {k: v.solution_value() for k, v in x_vars.items()}
    y_frac = {c: y_vars[c].solution_value() for c in C}

    # reduced cost is available for simplex-based LP
    rc_y = {c: y_vars[c].ReducedCost() for c in C}


    return dict(z_LP=z_LP, x_frac=x_frac, y_frac=y_frac, rc_y=rc_y)
