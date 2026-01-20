# runner.py
# All-in-one experiment runner for IterLP2 thesis evaluation.
#
# Supports:
#   - Synthetic graphs: ER / Random Regular / Barabasi-Albert / Watts-Strogatz / Random Geometric
#   - DIMACS .col parsing
#   - Real-world edge-list parsing
#   - Baselines: DSATUR, Smallest-Last (SLO)
#   - IterLP2 + ablations (disable side/pack/ub1 accelerators via monkey-patch)
#   - Parameter sweeps (perturb_y, strong_margin, restarts, max_fix_per_round)
#   - Optional convergence trace extraction from stdout lines:
#       [Round i] zLP=... | ceil(zLP)=... | K=... | UB=... | t=...s
#
# Outputs:
#   - A raw per-run CSV (one row per algorithm run)
#   - Optional per-run trace CSV (round-by-round)
#   - Optional summary CSV (aggregate per instance+algo)
#
# Requirements: python>=3.9, networkx, ortools (already in your repo's README)  :contentReference[oaicite:14]{index=14}

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import os
import re
import sys
import time
import threading
import signal
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

# Project imports (same style as runner_basic.py)  :contentReference[oaicite:15]{index=15}
from graph.dsatur import dsatur_coloring
from graph.slo import smallest_last_coloring
from graph.verify import verify_coloring
from graph.clique import greedy_max_clique
from driver.iterate_lp import run_iterative_lp_v2

# We'll monkey-patch these inside driver.iterate_lp for ablations.
import driver.iterate_lp as itlp  # type: ignore


class TeeParser(io.TextIOBase):
    """
    把 iterlp 的 stdout/stderr 同时：
    1) 原样输出到终端（不影响你看日志）
    2) 按行解析，提取 UB、round 等信息
    """
    def __init__(self, real_stream, on_line):
        self.real = real_stream
        self.on_line = on_line
        self.buf = ""

    def write(self, s):
        self.real.write(s)
        self.real.flush()
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            try:
                self.on_line(line)
            except Exception:
                pass
        return len(s)

    def flush(self):
        self.real.flush()

# --------------------------
# Data structures
# --------------------------

@dataclass(frozen=True)
class Instance:
    name: str
    family: str
    meta: Dict[str, Any]
    graph: nx.Graph


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    kind: str  # "baseline" | "iterlp2"
    params: Dict[str, Any]


# --------------------------
# Utilities
# --------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def atomic_write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    """
    原子重写 CSV：写到 .tmp -> flush+fsync -> os.replace
    目的：即使被 kill，也尽量保证磁盘上始终有一个“完整可读”的 CSV。
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    ensure_dir(path.parent)

    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

def graph_basic_stats(G: nx.Graph) -> Dict[str, Any]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dens = (2.0 * m / (n * (n - 1))) if n >= 2 else 0.0
    avg_deg = (2.0 * m / n) if n >= 1 else 0.0
    return {"n": n, "m": m, "density": dens, "avg_deg": avg_deg}


def compact_node_labels(G: nx.Graph) -> nx.Graph:
    # Ensure nodes are 0..n-1 to keep output consistent across loaders.
    return nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")


# --------------------------
# Loaders: DIMACS .col and edge-list
# --------------------------

_DIMACS_P_LINE = re.compile(r"^\s*p\s+(\w+)\s+(\d+)\s+(\d+)\s*$", re.IGNORECASE)
_DIMACS_E_LINE = re.compile(r"^\s*e\s+(\d+)\s+(\d+)\s*$", re.IGNORECASE)

def load_dimacs_col(path: Path) -> nx.Graph:
    """
    DIMACS .col format (typical):
      c comment
      p edge <n> <m>
      e u v
    Nodes are usually 1-based; we convert to 0-based.
    """
    n_decl = None
    edges: List[Tuple[int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            mp = _DIMACS_P_LINE.match(line)
            if mp:
                _ptype, n_s, _m_s = mp.groups()
                n_decl = int(n_s)
                continue
            me = _DIMACS_E_LINE.match(line)
            if me:
                u_s, v_s = me.groups()
                u = int(u_s) - 1
                v = int(v_s) - 1
                if u != v:
                    edges.append((u, v))
                continue

    if n_decl is None:
        # Fallback: infer n from max node id
        mx = -1
        for (u, v) in edges:
            mx = max(mx, u, v)
        n_decl = mx + 1

    G = nx.Graph()
    G.add_nodes_from(range(n_decl))
    G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    return compact_node_labels(G)


def load_edgelist_txt(path: Path) -> nx.Graph:
    """
    Simple undirected edge list:
      u v
    Nodes can be arbitrary integers; we will relabel to 0..n-1.
    Lines starting with # or c are ignored.
    """
    edges: List[Tuple[int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("c"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except Exception:
                continue
            if u != v:
                edges.append((u, v))
    G = nx.Graph()
    G.add_edges_from(edges)
    G.remove_edges_from(nx.selfloop_edges(G))
    return compact_node_labels(G)


# --------------------------
# Synthetic instance generator
# --------------------------

def gen_er_instances(ns: Sequence[int], ps: Sequence[float], seeds: Sequence[int]) -> List[Instance]:
    out: List[Instance] = []
    for n in ns:
        for p in ps:
            for sd in seeds:
                name = f"ER_n{n}_p{p:g}_gseed{sd}"
                G = nx.erdos_renyi_graph(n=n, p=p, seed=sd)
                out.append(Instance(name=name, family="synthetic_er", meta={"n": n, "p": p, "graph_seed": sd}, graph=G))
    return out


def gen_rr_instances(ns: Sequence[int], ds: Sequence[int], seeds: Sequence[int]) -> List[Instance]:
    out: List[Instance] = []
    for n in ns:
        for d in ds:
            if (n * d) % 2 != 0 or d >= n:
                # invalid random-regular parameters
                continue
            for sd in seeds:
                name = f"RR_n{n}_d{d}_gseed{sd}"
                G = nx.random_regular_graph(d=d, n=n, seed=sd)
                out.append(Instance(name=name, family="synthetic_rr", meta={"n": n, "d": d, "graph_seed": sd}, graph=G))
    return out


def gen_ba_instances(ns: Sequence[int], ms: Sequence[int], seeds: Sequence[int]) -> List[Instance]:
    out: List[Instance] = []
    for n in ns:
        for m in ms:
            if m <= 0 or m >= n:
                continue
            for sd in seeds:
                name = f"BA_n{n}_m{m}_gseed{sd}"
                G = nx.barabasi_albert_graph(n=n, m=m, seed=sd)
                out.append(Instance(name=name, family="synthetic_ba", meta={"n": n, "m": m, "graph_seed": sd}, graph=G))
    return out


def gen_ws_instances(ns: Sequence[int], ks: Sequence[int], betas: Sequence[float], seeds: Sequence[int]) -> List[Instance]:
    out: List[Instance] = []
    for n in ns:
        for k in ks:
            if k <= 0 or k >= n or (k % 2) != 0:
                continue
            for beta in betas:
                for sd in seeds:
                    name = f"WS_n{n}_k{k}_b{beta:g}_gseed{sd}"
                    G = nx.watts_strogatz_graph(n=n, k=k, p=beta, seed=sd)
                    out.append(Instance(
                        name=name, family="synthetic_ws",
                        meta={"n": n, "k": k, "beta": beta, "graph_seed": sd},
                        graph=G
                    ))
    return out


def gen_rgg_instances(ns: Sequence[int], radii: Sequence[float], seeds: Sequence[int]) -> List[Instance]:
    out: List[Instance] = []
    for n in ns:
        for r in radii:
            for sd in seeds:
                name = f"RGG_n{n}_r{r:g}_gseed{sd}"
                G = nx.random_geometric_graph(n=n, radius=r, seed=sd)
                out.append(Instance(name=name, family="synthetic_rgg", meta={"n": n, "radius": r, "graph_seed": sd}, graph=G))
    return out


# --------------------------
# Algorithms
# --------------------------

def ub_from_coloring(coloring: Dict[int, int]) -> int:
    return len(set(coloring.values())) if coloring else 0


def run_baseline(G: nx.Graph, algo: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if algo == "dsatur":
        col = dsatur_coloring(G)
    elif algo == "slo":
        col = smallest_last_coloring(G)
    else:
        raise ValueError(f"Unknown baseline algo: {algo}")

    UB = ub_from_coloring(col)
    rep = verify_coloring(G, col, allowed_colors=list(range(max(1, UB))))
    dt = time.perf_counter() - t0

    # LB via greedy max clique (same as in your other runner)  :contentReference[oaicite:16]{index=16}
    LB = len(greedy_max_clique(G))

    return {
        "LB": LB,
        "UB": rep["num_used_colors"],
        "gap": rep["num_used_colors"] - LB,
        "feasible": rep["feasible"],
        "conflicts": rep["num_conflicts"],
        "runtime_sec": dt,
        "iters": 0,
        "stop_reason": "baseline",
        "best_time_sec": dt,
        "best_round": 0,
    }


# ---- Ablations via monkey-patch (no changes to your codebase) ----

@dataclass
class PatchState:
    orig_side: Any
    orig_pack: Any
    orig_ub1: Any

def patch_iterlp2(variant: str) -> PatchState:
    """
    variant:
      - "full" (no patch)
      - "no_side"
      - "no_pack"
      - "no_ub1"
      - "no_accels" (disable all three)
    """
    st = PatchState(
        orig_side=getattr(itlp, "try_side_rounding"),
        orig_pack=getattr(itlp, "try_lp_guided_pack"),
        orig_ub1=getattr(itlp, "try_graph_ub1_greedy"),
    )

    def _no_side(G, x_frac, y_frac, K, UB, **kwargs):
        return UB, {}, False

    def _no_pack(G, best_coloring, x_frac, K, UB, **kwargs):
        return UB, {}, False

    def _no_ub1(G, best_coloring, UB, **kwargs):
        return UB, {}, False

    if variant == "full":
        return st
    if variant in ("no_side", "no_accels"):
        setattr(itlp, "try_side_rounding", _no_side)
    if variant in ("no_pack", "no_accels"):
        setattr(itlp, "try_lp_guided_pack", _no_pack)
    if variant in ("no_ub1", "no_accels"):
        setattr(itlp, "try_graph_ub1_greedy", _no_ub1)

    return st


def unpatch_iterlp2(st: PatchState) -> None:
    setattr(itlp, "try_side_rounding", st.orig_side)
    setattr(itlp, "try_lp_guided_pack", st.orig_pack)
    setattr(itlp, "try_graph_ub1_greedy", st.orig_ub1)


# ---- Convergence trace parsing (from stdout) ----
# Example line in your logs:  :contentReference[oaicite:17]{index=17}
# [Round 1] zLP=3.000000 | ceil(zLP)=3 | K=8 | UB=8 | t=0.53s
_ROUND_RE = re.compile(
    r"\[Round\s+(\d+)\]\s+zLP=([0-9.]+)\s+\|\s+ceil\(zLP\)=([0-9.]+)\s+\|\s+K=(\d+)\s+\|\s+UB=(\d+)\s+\|\s+t=([0-9.]+)s"
)

def extract_trace(stdout_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in stdout_text.splitlines():
        m = _ROUND_RE.search(line)
        if not m:
            continue
        rid, zlp, czlp, K, UB, tsec = m.groups()
        rows.append({
            "round": int(rid),
            "zLP": float(zlp),
            "ceil_zLP": float(czlp),
            "K": int(K),
            "UB": int(UB),
            "t_sec": float(tsec),
        })
    return rows


def run_iterlp2(
    G: nx.Graph,
    *,
    time_limit: int,
    init_heuristic: str,
    fix_policy: str,
    strong_margin: float,
    max_fix_per_round: int,
    restarts: int,
    perturb_y: float,
    algo_seed: int,
    ablation: str,
    save_trace: bool,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Dict[str, Any], Optional[str], List[Dict[str, Any]]]:
    """
    Returns:
      (metrics_row, captured_stdout_or_None, trace_rows)
    """
    # Apply ablation patch
    st = patch_iterlp2(ablation)
    try:
        t0 = time.perf_counter()
        captured = None
        trace_rows: List[Dict[str, Any]] = []

        if save_trace:
            buf = io.StringIO()
            with redirect_stdout(buf):
                res = run_iterative_lp_v2(
                    G,
                    time_limit_sec=time_limit,
                    verbose=True,  # must be True to emit [Round i] lines for parsing
                    init_heuristic=init_heuristic,
                    fix_policy=fix_policy,
                    strong_margin=strong_margin,
                    max_fix_per_round=max_fix_per_round,
                    restarts=restarts,
                    perturb_y=perturb_y,
                    enable_visualization=False,
                    algo_seed=algo_seed,
                )
            captured = buf.getvalue()
            trace_rows = extract_trace(captured)

        elif on_progress is not None:
            # live-update：不保存 trace，但要 verbose 输出日志供解析
            start_wall = time.time()

            INIT_RE = re.compile(r"\[Init\].*UB\(init:[^)]+\)=(\d+)")
            IMPROVE_RE = re.compile(r"\[Rounding/Local\]\s+improved UB\s*->\s*(\d+)")
            ROUND_RE = re.compile(r"\[Round\s+(\d+)\].*?\bUB=(\d+)\b.*?\bt=([0-9.]+)s")

            def _emit(round_id: int, ub: int, tsec: float) -> None:
                on_progress({"round": int(round_id), "UB": int(ub), "t_sec": float(tsec), "K": -1})

            def _on_line(line: str) -> None:
                # 1) 尽早拿到初始 UB（不等第一轮 LP）
                m0 = INIT_RE.search(line)
                if m0:
                    ub0 = int(m0.group(1))
                    _emit(0, ub0, time.time() - start_wall)
                    return

                # 2) 解析每轮信息（Round 行在 inc.solve() 结束后才会出现）
                m1 = ROUND_RE.search(line)
                if m1:
                    rid = int(m1.group(1))
                    ub = int(m1.group(2))
                    tsec = float(m1.group(3))  # 直接用算法自己打印的 elapsed
                    _emit(rid, ub, tsec)
                    return

                # 3) 若本轮本地搜索提升 UB，会有更早的 improved 行
                m2 = IMPROVE_RE.search(line)
                if m2:
                    ub = int(m2.group(1))
                    _emit(-1, ub, time.time() - start_wall)
                    return

            tee = TeeParser(sys.stdout, _on_line)
            with redirect_stdout(tee), redirect_stderr(tee):
                res = run_iterative_lp_v2(
                    G,
                    time_limit_sec=time_limit,
                    verbose=True,  # 必须 True 才会打印 [Init]/[Round] 行
                    init_heuristic=init_heuristic,
                    fix_policy=fix_policy,
                    strong_margin=strong_margin,
                    max_fix_per_round=max_fix_per_round,
                    restarts=restarts,
                    perturb_y=perturb_y,
                    enable_visualization=False,
                    algo_seed=algo_seed,
                )

        else:
            res = run_iterative_lp_v2(
                G,
                time_limit_sec=time_limit,
                verbose=False,
                init_heuristic=init_heuristic,
                fix_policy=fix_policy,
                strong_margin=strong_margin,
                max_fix_per_round=max_fix_per_round,
                restarts=restarts,
                perturb_y=perturb_y,
                enable_visualization=False,
                algo_seed=algo_seed,
            )


        dt = time.perf_counter() - t0

        row = {
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
        return row, captured, trace_rows
    finally:
        unpatch_iterlp2(st)


# --------------------------
# Suites / presets
# --------------------------

def build_instances(args: argparse.Namespace) -> List[Instance]:
    inst: List[Instance] = []

    graph_seeds = parse_int_list(args.graph_seeds)

    if args.suite in ("quick", "all"):
        # Minimal sanity checks (also useful for thesis “toy examples” table)
        inst.extend([
            Instance("K6", "toy", {"type": "complete", "n": 6}, nx.complete_graph(6)),
            Instance("C9", "toy", {"type": "cycle", "n": 9}, nx.cycle_graph(9)),
        ])

    if args.suite in ("synthetic", "all"):
        # Default thesis-grade synthetic sweep (you can tighten if runtime is too high)
        er_ns = parse_int_list(args.er_ns)
        er_ps = parse_float_list(args.er_ps)
        rr_ns = parse_int_list(args.rr_ns)
        rr_ds = parse_int_list(args.rr_ds)

        inst.extend(gen_er_instances(er_ns, er_ps, graph_seeds))
        inst.extend(gen_rr_instances(rr_ns, rr_ds, graph_seeds))

        if args.enable_more_synth:
            ba_ns = parse_int_list(args.ba_ns)
            ba_ms = parse_int_list(args.ba_ms)
            ws_ns = parse_int_list(args.ws_ns)
            ws_ks = parse_int_list(args.ws_ks)
            ws_bs = parse_float_list(args.ws_bs)
            rgg_ns = parse_int_list(args.rgg_ns)
            rgg_rs = parse_float_list(args.rgg_rs)

            inst.extend(gen_ba_instances(ba_ns, ba_ms, graph_seeds))
            inst.extend(gen_ws_instances(ws_ns, ws_ks, ws_bs, graph_seeds))
            inst.extend(gen_rgg_instances(rgg_ns, rgg_rs, graph_seeds))

    if args.suite in ("dimacs", "all"):
        ddir = Path(args.dimacs_dir)
        if ddir.exists():
            for p in sorted(ddir.glob("*.col")):
                G = load_dimacs_col(p)
                inst.append(Instance(p.stem, "dimacs", {"path": str(p)}, G))
        else:
            print(f"[WARN] DIMACS dir not found: {ddir}", file=sys.stderr)

    if args.suite in ("real", "all"):
        rdir = Path(args.edgelist_dir)
        if rdir.exists():
            for p in sorted(rdir.glob("*.txt")):
                G = load_edgelist_txt(p)
                inst.append(Instance(p.stem, "real", {"path": str(p)}, G))
        else:
            print(f"[WARN] Real-world edgelist dir not found: {rdir}", file=sys.stderr)

    # Final cleanup
    cleaned: List[Instance] = []
    for x in inst:
        G = compact_node_labels(x.graph)
        cleaned.append(Instance(x.name, x.family, dict(x.meta), G))

    # Optional cap
    if args.max_instances > 0:
        cleaned = cleaned[: args.max_instances]

    return cleaned


def build_algos(args: argparse.Namespace) -> List[AlgoSpec]:
    algos: List[AlgoSpec] = []
    for a in args.algos.split(","):
        a = a.strip()
        if not a:
            continue
        if a in ("dsatur", "slo"):
            algos.append(AlgoSpec(name=a, kind="baseline", params={}))
            continue

        # IterLP2 variants
        if a.startswith("iterlp2"):
            # allowed:
            # iterlp2_full, iterlp2_no_side, iterlp2_no_pack, iterlp2_no_ub1, iterlp2_no_accels
            ab = "full"
            if a.endswith("_no_side"):
                ab = "no_side"
            elif a.endswith("_no_pack"):
                ab = "no_pack"
            elif a.endswith("_no_ub1"):
                ab = "no_ub1"
            elif a.endswith("_no_accels"):
                ab = "no_accels"

            algos.append(AlgoSpec(
                name=a,
                kind="iterlp2",
                params={
                    "ablation": ab,
                    "init_heuristic": args.init_heuristic,
                    "fix_policy": args.fix_policy,
                }
            ))
            continue

        raise ValueError(f"Unknown algo spec: {a}")

    return algos


# --------------------------
# Aggregation
# --------------------------

def aggregate_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate per (instance, algo) over different algo_seeds (and potentially multiple runs).
    """
    key = lambda r: (r["instance"], r["algo"])
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(key(r), []).append(r)

    out: List[Dict[str, Any]] = []
    for (inst, algo), xs in sorted(buckets.items()):
        UBs = [int(x["UB"]) for x in xs if str(x.get("UB", "")).isdigit()]
        times = [float(x["runtime_sec"]) for x in xs if x.get("runtime_sec") not in ("", None)]
        best_times = []
        for x in xs:
            bt = x.get("best_time_sec", "")
            try:
                best_times.append(float(bt))
            except Exception:
                pass

        def _mean(v: List[float]) -> float:
            return sum(v) / len(v) if v else float("nan")

        out.append({
            "instance": inst,
            "algo": algo,
            "runs": len(xs),
            "UB_min": min(UBs) if UBs else "",
            "UB_mean": _mean([float(u) for u in UBs]) if UBs else "",
            "runtime_mean_sec": _mean(times) if times else "",
            "best_time_mean_sec": _mean(best_times) if best_times else "",
        })
    return out


# --------------------------
# Parsing helpers
# --------------------------

def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# --------------------------
# Main
# --------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="synthetic", choices=["quick", "synthetic", "dimacs", "real", "all"])
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--out-csv", default="results_runs.csv")
    ap.add_argument("--summary-csv", default="results_summary.csv")
    ap.add_argument("--time-limit", type=int, default=60)
    ap.add_argument("--live-update", type=int, default=0)     # 1=enable periodic atomic CSV rewrite
    ap.add_argument("--live-every", type=float, default=5.0)  # seconds between flushes

    # Seeds
    ap.add_argument("--graph-seeds", default="0,1,2,3,4")
    ap.add_argument("--algo-seeds", default="0,1,2,3,4")

    # Algorithms
    ap.add_argument("--algos", default="dsatur,slo,iterlp2_full")
    ap.add_argument("--init-heuristic", default="smallest_last", choices=["dsatur", "smallest_last"])
    ap.add_argument("--fix-policy", default="prefix_shrink+rounded_support")
    ap.add_argument("--strong-margin", type=float, default=0.25)
    ap.add_argument("--max-fix-per-round", type=int, default=50)
    ap.add_argument("--restarts", type=int, default=48)
    ap.add_argument("--perturb-y", type=float, default=1e-6)

    # Trace
    ap.add_argument("--save-trace", type=int, default=0)
    ap.add_argument("--trace-dir", default="results/traces")

    # Synthetic ranges
    ap.add_argument("--er-ns", default="60,100,150,200")
    ap.add_argument("--er-ps", default="0.04,0.06,0.08,0.10")
    ap.add_argument("--rr-ns", default="100,200")
    ap.add_argument("--rr-ds", default="6,10,14,18")

    ap.add_argument("--enable-more-synth", type=int, default=0)
    ap.add_argument("--ba-ns", default="100,200")
    ap.add_argument("--ba-ms", default="2,4,6")
    ap.add_argument("--ws-ns", default="100,200")
    ap.add_argument("--ws-ks", default="6,10,14")
    ap.add_argument("--ws-bs", default="0.05,0.10,0.20")
    ap.add_argument("--rgg-ns", default="100,200")
    ap.add_argument("--rgg-rs", default="0.12,0.15,0.18")

    # DIMACS/real dirs
    ap.add_argument("--dimacs-dir", default="data/dimacs")
    ap.add_argument("--edgelist-dir", default="data/real")

    # Caps
    ap.add_argument("--max-instances", type=int, default=0)  # 0 = no cap

    # Parameter sweep mode (optional)
    ap.add_argument("--sweep", type=int, default=0)
    ap.add_argument("--sweep-perturb-y", default="0,1e-9,1e-6,1e-3")
    ap.add_argument("--sweep-strong-margin", default="0.10,0.15,0.20,0.25,0.30")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    trace_dir = Path(args.trace_dir)
    if args.save_trace:
        ensure_dir(trace_dir)

    instances = build_instances(args)
    algos = build_algos(args)

    algo_seeds = parse_int_list(args.algo_seeds)
    if not algo_seeds:
        algo_seeds = [0]

    # Prepare CSV rows
    run_rows: List[Dict[str, Any]] = []

    # Sweep grid (if enabled)
    sweep_perturb = parse_float_list(args.sweep_perturb_y)
    sweep_margin = parse_float_list(args.sweep_strong_margin)

    def iter_param_configs() -> Iterable[Tuple[float, float]]:
        if not args.sweep:
            yield (args.perturb_y, args.strong_margin)
        else:
            for py in sweep_perturb:
                for sm in sweep_margin:
                    yield (py, sm)

    print(f"[Runner] instances={len(instances)} algos={len(algos)} algo_seeds={algo_seeds} time_limit={args.time_limit}s")

    # 预定义 runs.csv 的稳定 header（避免你原来“union keys”导致 live 写时变来变去）
    RUN_FIELDS = [
        "instance","family","n","m","density","avg_deg",
        "algo","algo_seed","time_limit_sec",
        "init_heuristic","fix_policy","strong_margin","max_fix_per_round","restarts","perturb_y",
        "ablation",
        "LB","UB","gap","feasible","conflicts","runtime_sec","iters","stop_reason","best_time_sec","best_round",
        "status","last_update_ts","error",
    ]
    SUM_FIELDS = ["instance","algo","runs","UB_min","UB_mean","runtime_mean_sec","best_time_mean_sec"]

    out_csv_path = out_dir / args.out_csv
    out_sum_path = out_dir / args.summary_csv

    csv_lock = threading.Lock()

    def live_flush_all() -> None:
        with csv_lock:
            atomic_write_csv(out_csv_path, RUN_FIELDS, run_rows)
            summ_rows = aggregate_summary(run_rows)
            atomic_write_csv(out_sum_path, SUM_FIELDS, summ_rows)


    for inst in instances:
        G = inst.graph
        stats = graph_basic_stats(G)
        for algo in algos:
            for (py, sm) in iter_param_configs():
                for aseed in algo_seeds:
                    row_base = {
                        "instance": inst.name,
                        "family": inst.family,
                        "n": stats["n"],
                        "m": stats["m"],
                        "density": stats["density"],
                        "avg_deg": stats["avg_deg"],
                        "algo": algo.name,
                        "algo_seed": aseed,
                        "time_limit_sec": args.time_limit,
                        "init_heuristic": args.init_heuristic if algo.kind == "iterlp2" else "",
                        "fix_policy": args.fix_policy if algo.kind == "iterlp2" else "",
                        "strong_margin": sm if algo.kind == "iterlp2" else "",
                        "max_fix_per_round": args.max_fix_per_round if algo.kind == "iterlp2" else "",
                        "restarts": args.restarts if algo.kind == "iterlp2" else "",
                        "perturb_y": py if algo.kind == "iterlp2" else "",
                    }
                    # --- 创建占位行：即使后面被杀，CSV 至少存在且有 running 状态 ---
                    run_row = dict(row_base)
                    run_row.update({
                        "ablation": "",
                        "LB": "", "UB": "", "gap": "", "feasible": False, "conflicts": "",
                        "runtime_sec": 0.0, "iters": "", "stop_reason": "",
                        "best_time_sec": "", "best_round": "",
                        "status": "running",
                        "last_update_ts": int(time.time()),
                        "error": "",
                    })
                    if algo.kind == "iterlp2":
                        run_row["ablation"] = algo.params.get("ablation", "full")

                    run_rows.append(run_row)
                    # 当前行引用（后面实时更新）
                    cur = run_rows[-1]

                    if args.live_update:
                        live_flush_all()
                    try:
                        if algo.kind == "baseline":
                        
                            met = run_baseline(G, algo.name)
                            cur.update(met)
                            cur["status"] = "done"
                            cur["last_update_ts"] = int(time.time())
                            run_rows[-1] = cur
                            if args.live_update:
                                live_flush_all()

                        elif algo.kind == "iterlp2":
                            ablation = algo.params.get("ablation", "full")

                            t0 = time.perf_counter()
                            stop_evt = threading.Event()
                            state = {
                                "best_ub": None,
                                "best_round": "",
                                "best_time_sec": "",
                                "last_ub": None,
                            }

                            def on_progress(p: Dict[str, Any]) -> None:
                                # p: {"round":..., "UB":..., "t_sec":...}
                                ub = int(p["UB"])
                                rid = int(p["round"])
                                tsec = float(p["t_sec"])

                                state["last_ub"] = ub
                                # UB 越小越好（颜色数）
                                if state["best_ub"] is None or ub < state["best_ub"]:
                                    state["best_ub"] = ub
                                    state["best_round"] = rid
                                    state["best_time_sec"] = tsec

                                # 只更新内存里的 cur，不要每行日志都写盘（写盘交给 ticker）
                                cur["UB"] = ub
                                cur["best_round"] = state["best_round"]
                                cur["best_time_sec"] = state["best_time_sec"]
                                cur["last_update_ts"] = int(time.time())

                            def ticker():
                                while not stop_evt.wait(args.live_every):
                                    cur["runtime_sec"] = time.perf_counter() - t0
                                    cur["status"] = "running"
                                    cur["last_update_ts"] = int(time.time())
                                    live_flush_all()

                            th = None
                            if args.live_update:
                                th = threading.Thread(target=ticker, daemon=True)
                                th.start()

                            try:
                                met, captured, trace_rows = run_iterlp2(
                                    G,
                                    time_limit=args.time_limit,
                                    init_heuristic=args.init_heuristic,
                                    fix_policy=args.fix_policy,
                                    strong_margin=float(sm),
                                    max_fix_per_round=int(args.max_fix_per_round),
                                    restarts=int(args.restarts),
                                    perturb_y=float(py),
                                    algo_seed=int(aseed),
                                    ablation=str(ablation),
                                    save_trace=bool(args.save_trace),
                                    on_progress=on_progress if args.live_update else None,
                                )
                                cur.update(met)
                                cur["ablation"] = ablation
                                cur["status"] = "done"
                                cur["last_update_ts"] = int(time.time())
                                if args.live_update:
                                    live_flush_all()

                                if args.save_trace and trace_rows:
                                    tfile = trace_dir / f"{inst.name}__{algo.name}__seed{aseed}__py{py:g}__sm{sm:g}.trace.csv"
                                    with tfile.open("w", newline="", encoding="utf-8") as f:
                                        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
                                        w.writeheader()
                                        w.writerows(trace_rows)

                            finally:
                                stop_evt.set()
                                if th is not None:
                                    th.join(timeout=1.0)


                    except Exception as e:
                        cur.update({
                            "LB": "",
                            "UB": cur.get("UB", ""),
                            "gap": "",
                            "feasible": False,
                            "conflicts": "",
                            "runtime_sec": cur.get("runtime_sec", ""),
                            "iters": "",
                            "stop_reason": f"ERROR:{type(e).__name__}",
                            "best_time_sec": cur.get("best_time_sec", ""),
                            "best_round": cur.get("best_round", ""),
                            "status": "error",
                            "error": str(e)[:300],
                            "last_update_ts": int(time.time()),
                        })
                        if args.live_update:
                            live_flush_all()


    # Final flush / write outputs
    if args.live_update:
        # live 模式：全程已原子重写 runs/summary，这里只需要再 flush 一次
        live_flush_all()
        print(f"[Runner] wrote runs -> {out_csv_path}")
        print(f"[Runner] wrote summary -> {out_sum_path}")
    else:
        # 非 live：沿用原来的一次性写入
        out_csv = out_dir / args.out_csv
        if run_rows:
            keys: List[str] = []
            seen = set()
            for r in run_rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)

            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(run_rows)

        summ_rows = aggregate_summary(run_rows)
        out_summ = out_dir / args.summary_csv
        if summ_rows:
            with out_summ.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summ_rows[0].keys()))
                w.writeheader()
                w.writerows(summ_rows)

        print(f"[Runner] wrote runs -> {out_csv}")
        print(f"[Runner] wrote summary -> {out_summ}")

    if args.save_trace:
        print(f"[Runner] traces -> {trace_dir}")

if __name__ == "__main__":
    main()
