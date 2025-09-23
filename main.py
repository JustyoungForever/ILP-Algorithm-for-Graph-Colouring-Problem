from driver.iterate_lp import run_iterative_lp
from graph.loader import load_demo_graph
from graph.verify import print_check_summary, verify_coloring

# baseline:DSATUR
from graph.dsatur import dsatur_coloring

import time

if __name__ == "__main__":
    G = load_demo_graph()  # load the small  graph

    #baseline: DSATUR
    t0 = time.time()
    ds_col = dsatur_coloring(G)
    t_ds = time.time() - t0
    UB_ds = len(set(ds_col.values()))
    rep_ds = verify_coloring(G, ds_col, allowed_colors=list(range(UB_ds)))

    print("=== DSATUR Baseline ===")
    print(f"UB (colors used): {UB_ds}")
    print(f"Time (s): {t_ds:.6f}")
    print_check_summary(rep_ds, prefix="[DSATUR] ")

    # iterative LP heuristic
    t1 = time.time()
    result = run_iterative_lp(
        G,
        time_limit_sec=6000,
        max_rounds=200,
        stall_rounds=10,
        min_rounds=5,
        verbose=True,
    )
    t_lp = time.time() - t1

    print("\n=== Final Result (Iterative LP) ===")
    print(f"LB (clique size): {result['LB']}")
    print(f"UB (colors used): {result['UB']}")
    print(f"Iterations: {result['iters']}")
    print(f"Stopped: {result['stop_reason']}")
    print(f"Time (s): {t_lp:.6f}")
    print_check_summary(result['final_check'], prefix="[IterLP] ")

    # side-by-side comparison
    print("\n=== Comparison: DSATUR vs Iterative-LP ===")
    delta = UB_ds - result['UB']
    better = "Iterative-LP better" if delta > 0 else ("Tie" if delta == 0 else "DSATUR better")
    print(f"DSATUR colors = {UB_ds} | IterLP colors = {result['UB']}  -> {better} (Δ = {delta})")
    print(f"DSATUR time   = {t_ds:.6f}s | IterLP time   = {t_lp:.6f}s")
