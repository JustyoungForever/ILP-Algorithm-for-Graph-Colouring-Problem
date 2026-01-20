# experiments/smoke_test.py
import networkx as nx

from graph.dsatur import dsatur_coloring
from graph.slo import smallest_last_coloring
from graph.verify import verify_coloring
from driver.iterate_lp import run_iterative_lp_v2


def _ub(coloring: dict[int, int]) -> int:
    return len(set(coloring.values())) if coloring else 0


def run_one(G: nx.Graph, name: str) -> None:
    print(f"\n=== {name} === |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")

    col0 = dsatur_coloring(G)
    UB0 = _ub(col0)
    rep0 = verify_coloring(G, col0, allowed_colors=list(range(UB0)))
    print(f"[DSATUR] feasible={rep0['feasible']} UB={rep0['num_used_colors']} conflicts={rep0['num_conflicts']}")

    col1 = smallest_last_coloring(G)
    UB1 = _ub(col1)
    rep1 = verify_coloring(G, col1, allowed_colors=list(range(UB1)))
    print(f"[SLO]    feasible={rep1['feasible']} UB={rep1['num_used_colors']} conflicts={rep1['num_conflicts']}")

    res = run_iterative_lp_v2(
        G,
        time_limit_sec=3,
        verbose=False,
        init_heuristic="smallest_last",
        fix_policy="prefix_shrink+rounded_support",
        strong_margin=0.25,
        max_fix_per_round=20,
        restarts=16,
        perturb_y=1e-6,
        enable_visualization=False,
        viz_out_dir="visualisierung/picture",
        viz_layout_seed=42,
        # 先不传 seed（下一步我们会加）
    )
    print(f"[IterLP2] feasible={res['feasible']} LB={res['LB']} UB={res['UB']} iters={res['iters']} stop={res['stop_reason']}")


def main() -> None:
    run_one(nx.complete_graph(6), "K6")
    run_one(nx.cycle_graph(9), "C9")
    run_one(nx.erdos_renyi_graph(60, 0.06, seed=0), "ER(60,0.06)")


if __name__ == "__main__":
    main()
