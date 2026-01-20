# experiments/runner_basic.py
import csv
import time
import networkx as nx

from graph.dsatur import dsatur_coloring
from graph.slo import smallest_last_coloring
from graph.verify import verify_coloring
from graph.clique import greedy_max_clique
from driver.iterate_lp import run_iterative_lp_v2


def _ub(coloring: dict[int, int]) -> int:
    return len(set(coloring.values())) if coloring else 0


def run_baseline(G: nx.Graph, inst: str, algo: str) -> dict:
    t0 = time.time()
    if algo == "dsatur":
        col = dsatur_coloring(G)
    elif algo == "slo":
        col = smallest_last_coloring(G)
    else:
        raise ValueError(f"Unknown baseline algo: {algo}")

    UB = _ub(col)
    rep = verify_coloring(G, col, allowed_colors=list(range(UB)))

    return {
        "instance": inst,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "algo": algo,
        "LB": len(greedy_max_clique(G)),
        "UB": rep["num_used_colors"],
        "gap": rep["num_used_colors"] - len(greedy_max_clique(G)),
        "feasible": rep["feasible"],
        "conflicts": rep["num_conflicts"],
        "runtime_sec": time.time() - t0,
        "iters": 0,
        "stop_reason": "baseline",
        "best_time_sec": "",
        "best_round": "",
    }


def run_iterlp2(G: nx.Graph, inst: str, time_limit: int = 60) -> dict:
    t0 = time.time()
    res = run_iterative_lp_v2(
        G,
        time_limit_sec=time_limit,
        verbose=False,
        init_heuristic="smallest_last",
        fix_policy="prefix_shrink+rounded_support",
        strong_margin=0.25,
        max_fix_per_round=100,
        restarts=256,
        perturb_y=1e-6,
        enable_visualization=False,
        # viz_out_dir="visualisierung/picture",
        # viz_layout_seed=42,
        algo_seed=0,
    )
    dt = time.time() - t0

    return {
        "instance": inst,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "algo": "iterlp2",
        "LB": res["LB"],
        "UB": res["UB"],
        "gap": res["UB"] - res["LB"],
        "feasible": res["feasible"],
        "conflicts": res["final_check"]["num_conflicts"],
        "runtime_sec": dt,
        "iters": res["iters"],
        "stop_reason": res["stop_reason"],
        "best_time_sec": res.get("best_time_sec", ""),
        "best_round": res.get("best_round", ""),
    }


def main() -> None:
    instances = [
        ("K6", nx.complete_graph(6)),
        ("C9", nx.cycle_graph(9)),
        ("ER60_p006", nx.erdos_renyi_graph(60, 0.06, seed=0)),
        ("RR100_d10", nx.random_regular_graph(10, 100, seed=0)),
    ]

    rows = []
    for name, G in instances:
        rows.append(run_baseline(G, name, "dsatur"))
        rows.append(run_baseline(G, name, "slo"))
        rows.append(run_iterlp2(G, name, time_limit=60))

    out = "results_basic.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
