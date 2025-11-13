Function MAIN(args)
  Input:
    algo ∈ {dsatur, iterlp, iterlp2}; seed ∈ ℤ; time ∈ ℤ₊;
    init_heuristic ∈ {dsatur, smallest_last};
    fix_policy: string with tokens {prefix_shrink, strong_assign, rounded_support} joined by '+';
    strong_margin ∈ ℝ₊; max_fix_per_round ∈ ℤ₊;
    restarts ∈ ℤ₊; perturb_y ∈ ℝ₊;
    viz_out: path; viz_layout_seed ∈ ℤ
  Output:
    console_summary: fixed lines (parameter line, DSATUR baseline, two compare-lines, final recap)
    final_figure: PNG saved into viz_out (DSATUR 或 IterLP2 最终图)

  Steps:
    G ← load_demo_graph(seed)
    if algo == "dsatur" then
       col ← dsatur_coloring(G)
       rep ← verify_coloring(G, col, allowed_colors = 0..|colors(col)|-1)
       print DSATUR summary lines; visualize_coloring(G, col, step="Final-DSATUR", round_id=0, out_dir=viz_out, layout_seed=viz_layout_seed)
    else  // iterlp2
       // baseline
       ds_col ← dsatur_coloring(G); rep_ds ← verify_coloring(G, ds_col, allowed_colors=...)
       print [Main] line, then [Baseline/DSATUR] line
       // main solver
       res ← run_iterative_lp_v2(G, time_limit_sec=time, init_heuristic, fix_policy,
                                 strong_margin, max_fix_per_round, restarts, perturb_y,
                                 enable_visualization=True, viz_out_dir=viz_out, viz_layout_seed)
       print [Compare] lines from res and dsatur baseline
       visualize_coloring(G, res.coloring, step="Final-IterLP2", round_id=res.iters, out_dir=viz_out, layout_seed)
       print final recap from res
End
