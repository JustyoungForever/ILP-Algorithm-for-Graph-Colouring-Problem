# tests/smoke_tests.py
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
import networkx as nx
from driver.iterate_lp import run_iterative_lp
from graph.verify import verify_coloring

def as_int_labels(G):

    return nx.convert_node_labels_to_integers(G)

def run_and_check(G, expect_chi=None, name="Graph"):
    G = as_int_labels(G)
    res = run_iterative_lp(G, time_limit_sec=10, max_rounds=50, verbose=False)
    ok = res["feasible"]
    used = res["UB"]
    lb = res["LB"]

    assert ok, f"{name}: infeasible coloring"
    assert used >= lb, f"{name}: UB < LB ({used} < {lb})"

    if expect_chi is not None:
        assert used == expect_chi, f"{name}: expect χ={expect_chi}, got UB={used}"

    rep = verify_coloring(G, res["coloring"], allowed_colors=range(used))
    assert rep["feasible"], f"{name}: verify_coloring says infeasible"
    print(f"[PASS] {name:20s}  UB={used}  LB={lb}  feasible={ok}")
    return used

if __name__ == "__main__":

    run_and_check(nx.complete_graph(3), expect_chi=3, name="K3")
    run_and_check(nx.complete_graph(4), expect_chi=4, name="K4")
    run_and_check(nx.cycle_graph(4),    expect_chi=2, name="C4 (even cycle)")
    run_and_check(nx.cycle_graph(5),    expect_chi=3, name="C5 (odd cycle)")
    run_and_check(nx.complete_bipartite_graph(3,4), expect_chi=2, name="K3,4")
    run_and_check(nx.grid_2d_graph(5,5), expect_chi=2, name="Grid 5x5")
    run_and_check(nx.petersen_graph(),   expect_chi=3, name="Petersen")


    for i, p in enumerate([0.05, 0.08, 0.12], start=1):
        G = nx.erdos_renyi_graph(40, p, seed=i)
        run_and_check(G, expect_chi=None, name=f"ER(40,{p})")

    print("All smoke tests passed.")
