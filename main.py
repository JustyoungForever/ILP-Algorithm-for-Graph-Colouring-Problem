# main.py
import argparse, time
from graph.loader import load_demo_graph
from graph.verify import verify_coloring, print_check_summary
from graph.dsatur import dsatur_coloring
from driver.iterate_lp import run_iterative_lp_v2
from visualisierung.draw import visualize_coloring

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="iterlp2", choices=["dsatur", "iterlp", "iterlp2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", type=int, default=60)
    ap.add_argument("--init-heuristic", default="dsatur", choices=["dsatur","smallest_last"])
    ap.add_argument("--fix-policy", default="prefix_shrink+strong_assign")
    ap.add_argument("--strong-margin", type=float, default=0.25)
    ap.add_argument("--max-fix-per-round", type=int, default=50)
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--perturb-y", type=float, default=1e-6)
    ap.add_argument("--viz-out", default="visualisierung/picture")
    ap.add_argument("--viz-layout-seed", type=int, default=42)
    args = ap.parse_args()

    G = load_demo_graph(seed=args.seed)

    if args.algo == "dsatur":
        t0 = time.time()
        col = dsatur_coloring(G)
        t1 = time.time() - t0
        UB = len(set(col.values()))
        rep = verify_coloring(G, col, allowed_colors=list(range(UB)))
        print("=== DSATUR ===")
        print(f"UB={UB} time={t1:.4f}s")
        print_check_summary(rep, prefix="[DSATUR] ")
        visualize_coloring(
            G, col, step="Final-DSATUR", round_id=0,
            out_dir=args.viz_out, layout_seed=args.viz_layout_seed,
            only_colored=True,
            allowed_colors=None,
            conflict_nodes_fill_black=True,
        )

    else:
        if args.algo == "iterlp2":
            print("[Main] algo=iterlp2 | time=%ds | seed=%d | init=%s | fix-policy=%s | restarts=%d | perturb-y=%g"
                % (args.time, args.seed, args.init_heuristic, args.fix_policy, args.restarts, args.perturb_y))
            #  DSATUR baseline 
            t0 = time.time()
            ds_col = dsatur_coloring(G)
            ds_time = time.time() - t0
            UB_ds = len(set(ds_col.values()))
            rep_ds = verify_coloring(G, ds_col, allowed_colors=list(range(UB_ds)))
            print("[Baseline/DSATUR] colors=%d | time=%.4fs | feasible=%s | conflicts=%d"
                % (UB_ds, ds_time, rep_ds["feasible"], rep_ds["num_conflicts"]))

            #  run iterlp2 with timing 
            t1 = time.time()
            res = run_iterative_lp_v2(
                G,
                time_limit_sec=args.time,
                verbose=True,
                init_heuristic=args.init_heuristic,
                fix_policy=args.fix_policy,
                strong_margin=args.strong_margin,
                max_fix_per_round=args.max_fix_per_round,
                restarts=args.restarts,
                perturb_y=args.perturb_y,
                enable_visualization=True,          
                viz_out_dir=args.viz_out,
                viz_layout_seed=args.viz_layout_seed,
            )
            lp_time = time.time() - t1

            #  comparison with DSATUR
            UB_lp = res["UB"]
            delta = UB_ds - UB_lp
            verdict = "IterLP2 better" if delta > 0 else ("Tie" if delta == 0 else "DSATUR better")
            print("[Compare] DSATUR colors=%d vs IterLP2 colors=%d  → %s (Δ=%d)"
                % (UB_ds, UB_lp, verdict, delta))
            print("[Compare] DSATUR time=%.4fs | IterLP2 time=%.4fs | stop=%s"
                % (ds_time, lp_time, res["stop_reason"]))
            visualize_coloring(
                G, res["coloring"], step="Final-IterLP2", round_id=res["iters"],
                out_dir=args.viz_out, layout_seed=args.viz_layout_seed,
                only_colored=True,
                allowed_colors=None,
                conflict_nodes_fill_black=True,
            )
            # final short recap (keep your existing final print too)
            print("[Main] Done. stop_reason=%s | LB=%d | UB=%d | iters=%d"
                % (res["stop_reason"], res["LB"], res["UB"], res["iters"]))