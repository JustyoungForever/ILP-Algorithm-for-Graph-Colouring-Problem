#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment analysis framework for Graph Coloring results.

Input (single source of truth):
  results/_merged/master_runs.csv

Outputs (paper-ready):
  results/_analysis_out/tables/*.tex        LaTeX tables (booktabs)
  results/_analysis_out/figures/*.(png|pdf) figures for thesis
  results/_analysis_out/text/auto_findings.md  auto-generated findings draft
  results/_analysis_out/data/*.csv          derived tidy tables for debugging

Run:
  cd results
  python3 analysis/analyze_experiments.py
"""

from __future__ import annotations
from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------- Config -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent
INPUT_CSV = RESULTS_DIR / "_merged" / "master_runs.csv"
OUT_DIR   = RESULTS_DIR / "_analysis_out"

# 最小“论文对比”分组键（你现在参数基本默认时最稳）
GROUP_MIN = ["instance", "algo", "ablation", "time_limit_sec"]

# 论文正式版本建议分组键（当你开始 sweep 参数时启用）
GROUP_FULL = [
    "instance", "family", "n", "m", "density", "avg_deg",
    "algo", "ablation", "time_limit_sec",
    "init_heuristic", "fix_policy", "strong_margin", "max_fix_per_round", "restarts", "perturb_y",
]

# 现在先用 MIN，后续你只要把 USE_GROUP 改为 GROUP_FULL 即可
USE_GROUP = GROUP_MIN

# 期望列（缺失就补 NA，保证鲁棒）
EXPECTED_COLS = [
    "instance","family","n","m","density","avg_deg",
    "algo","algo_seed","time_limit_sec",
    "init_heuristic","fix_policy","strong_margin","max_fix_per_round",
    "restarts","perturb_y",
    "LB","UB","gap","feasible","conflicts",
    "runtime_sec","iters","stop_reason",
    "best_time_sec","best_round","ablation",
]

NUMERIC_COLS = [
    "n","m","density","avg_deg","algo_seed","time_limit_sec",
    "strong_margin","max_fix_per_round","restarts","perturb_y",
    "LB","UB","gap","conflicts","runtime_sec","iters","best_time_sec","best_round"
]

# 图像输出格式
FIG_FORMATS = ["png", "pdf"]

# ----------------------------------------------------------


def ensure_dirs():
    (OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "text").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data").mkdir(parents=True, exist_ok=True)


def to_bool_series(s: pd.Series) -> pd.Series:
    def conv(x):
        if pd.isna(x):
            return pd.NA
        t = str(x).strip().lower()
        if t in ("true","1","t","yes","y"):
            return True
        if t in ("false","0","f","no","n"):
            return False
        return pd.NA
    return s.apply(conv)


def load_master_runs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path.resolve()}")

    df = pd.read_csv(path, dtype="string", low_memory=False)

    # 补缺列
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # 类型清洗
    df["feasible"] = to_bool_series(df["feasible"])
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ablation：缺失时默认 baseline（你也可改成 "unknown"）
    df["ablation"] = df["ablation"].fillna("baseline")

    return df


def latex_table(df: pd.DataFrame, out_tex: Path, caption: str = "", label: str = ""):
    """
    Write a booktabs-style LaTeX table.
    """
    latex = df.to_latex(
        index=False,
        escape=True,
        longtable=False,
        caption=caption if caption else None,
        label=label if label else None,
        bold_rows=False,
        na_rep="",
        float_format=lambda x: f"{x:.3f}" if isinstance(x, float) and not math.isnan(x) else str(x),
    )
    # 强制 booktabs（pandas 默认就是 \toprule 等）
    out_tex.write_text(latex, encoding="utf-8")


def save_fig(name: str):
    for ext in FIG_FORMATS:
        plt.savefig(OUT_DIR / "figures" / f"{name}.{ext}", bbox_inches="tight", dpi=200)


def instance_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["instance","family","n","m","density","avg_deg"]
    tmp = df[cols].dropna(subset=["instance"]).copy()
    # 每个 instance 取第一条非空记录
    ov = (tmp.sort_values(cols)
            .groupby("instance", dropna=False, as_index=False)
            .first())
    # 规模排序（便于阅读）
    ov = ov.sort_values(["n","m","density"], ascending=[True, True, True])
    return ov


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    cov = (df.groupby(["instance","algo","ablation"], dropna=False)
             .size().reset_index(name="runs"))
    return cov.sort_values(["instance","algo","ablation"])


def agg_runs(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    def feasible_rate(x: pd.Series):
        x = x.dropna()
        if len(x) == 0:
            return np.nan
        return float((x == True).mean())

    agg = (df.groupby(group_cols, dropna=False)
             .agg(
                 runs=("UB","size"),
                 feasible_rate=("feasible", feasible_rate),
                 UB_min=("UB","min"),
                 UB_mean=("UB","mean"),
                 UB_std=("UB","std"),
                 runtime_mean_sec=("runtime_sec","mean"),
                 runtime_std_sec=("runtime_sec","std"),
                 best_time_mean_sec=("best_time_sec","mean"),
                 best_time_std_sec=("best_time_sec","std"),
             )
             .reset_index())
    return agg


def best_algo_win_count(summary: pd.DataFrame) -> pd.DataFrame:
    """
    For each instance (and ablation/time limit), count which algo achieves minimal UB_mean.
    """
    # 先按 instance+ablation+time_limit 汇总，否则不同配置混在一起
    key = ["instance","ablation","time_limit_sec"]
    use = summary.dropna(subset=["UB_mean"]).copy()
    # 每组找最小 UB_mean 的算法（若并列，全部计入）
    use["min_UB_mean_in_group"] = use.groupby(key)["UB_mean"].transform("min")
    win = use[use["UB_mean"] == use["min_UB_mean_in_group"]]
    out = (win.groupby(["algo","ablation","time_limit_sec"], dropna=False)
              .size().reset_index(name="win_count"))
    return out.sort_values(["ablation","time_limit_sec","win_count"], ascending=[True, True, False])


def stop_reason_dist(df: pd.DataFrame) -> pd.DataFrame:
    tmp = (df.groupby(["algo","ablation","stop_reason"], dropna=False)
             .size().reset_index(name="count"))
    tmp["share"] = tmp.groupby(["algo","ablation"], dropna=False)["count"].transform(lambda x: x / x.sum())
    return tmp.sort_values(["algo","ablation","count"], ascending=[True, True, False])


def plot_stop_reason(dist: pd.DataFrame):
    """
    Bar plot: stop_reason shares per algo (stacked bars).
    """
    # 只画 share，按 algo-ablation 聚合
    # 处理成 pivot：index=(algo,ablation), columns=stop_reason, values=share
    piv = dist.pivot_table(index=["algo","ablation"], columns="stop_reason", values="share", fill_value=0.0)
    piv = piv.sort_index()

    # stacked bar
    x = np.arange(len(piv.index))
    bottom = np.zeros(len(piv.index))

    plt.figure(figsize=(10, 4))
    for col in piv.columns:
        vals = piv[col].values.astype(float)
        plt.bar(x, vals, bottom=bottom, label=str(col))
        bottom += vals

    labels = [f"{a}|{b}" for (a,b) in piv.index]
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Share")
    plt.title("Stop reason distribution (share)")
    plt.legend(fontsize=8, ncol=2)
    save_fig("stop_reason_share")
    plt.close()


def plot_quality_box(summary: pd.DataFrame):
    """
    Box plot of UB_mean across instances per algo (separately per ablation if needed).
    """
    # 仅用 UB_mean
    tmp = summary.dropna(subset=["UB_mean"]).copy()
    # 你现在几乎都是 baseline/full，先按 ablation 做分面（最多两张图）
    for ab in sorted(tmp["ablation"].dropna().unique()):
        sub = tmp[tmp["ablation"] == ab]
        algos = sorted(sub["algo"].dropna().unique())
        data = [sub[sub["algo"] == a]["UB_mean"].astype(float).values for a in algos]

        plt.figure(figsize=(8, 4))
        plt.boxplot(data, labels=algos, showfliers=True)
        plt.ylabel("UB_mean")
        plt.title(f"UB_mean distribution across instances (ablation={ab})")
        plt.xticks(rotation=30, ha="right")
        save_fig(f"ub_mean_box_ablation_{ab}")
        plt.close()


def plot_runtime_vs_n(summary: pd.DataFrame, df_runs: pd.DataFrame):
    """
    Scatter: runtime_mean vs n per algo.
    Works even when summary doesn't contain n (e.g., GROUP_MIN).
    """
    tmp = summary.copy()

    # GROUP_MIN 下 summary 可能没有 n，这里从原始 df_runs 提取 instance->n 映射并 merge 回来
    if "n" not in tmp.columns:
        n_map = (df_runs[["instance", "n"]]
                 .dropna(subset=["instance", "n"])
                 .groupby("instance", as_index=False)
                 .first())
        tmp = tmp.merge(n_map, on="instance", how="left")

    # 防御性 dropna
    tmp = tmp.dropna(subset=["runtime_mean_sec", "n", "algo"]).copy()
    tmp["n"] = pd.to_numeric(tmp["n"], errors="coerce")
    tmp["runtime_mean_sec"] = pd.to_numeric(tmp["runtime_mean_sec"], errors="coerce")
    tmp = tmp.dropna(subset=["n", "runtime_mean_sec"])

    if len(tmp) == 0:
        return

    plt.figure(figsize=(8, 4))
    for a in sorted(tmp["algo"].dropna().unique()):
        sub = tmp[tmp["algo"] == a]
        plt.scatter(sub["n"].astype(float), sub["runtime_mean_sec"].astype(float), label=a, s=18)

    plt.xlabel("n (num vertices)")
    plt.ylabel("runtime_mean_sec")
    plt.title("Runtime vs graph size")
    plt.legend(fontsize=8)
    save_fig("runtime_vs_n_scatter")
    plt.close()


def plot_anytime_scatter(df: pd.DataFrame):
    """
    For IterLP-like algos: scatter best_time_sec vs runtime_sec.
    """
    tmp = df.dropna(subset=["best_time_sec","runtime_sec","algo"]).copy()
    # 只看 iterlp 相关（你也可按需放宽）
    mask = tmp["algo"].astype(str).str.contains("iterlp", case=False, na=False)
    tmp = tmp[mask]

    if len(tmp) == 0:
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(tmp["runtime_sec"].astype(float), tmp["best_time_sec"].astype(float), s=14)
    plt.xlabel("runtime_sec")
    plt.ylabel("best_time_sec")
    plt.title("Anytime behavior (IterLP runs)")
    save_fig("iterlp_anytime_scatter")
    plt.close()


def write_findings(summary: pd.DataFrame, wins: pd.DataFrame, stopdist: pd.DataFrame, out_md: Path):
    """
    Generate a draft of findings you can paste into thesis (then polish).
    """
    lines = []
    lines.append("# Auto findings draft\n")

    # 基本规模与覆盖
    n_inst = summary["instance"].nunique(dropna=True)
    n_alg = summary["algo"].nunique(dropna=True)
    lines.append(f"- 覆盖实例数（summary层面）: **{n_inst}**")
    lines.append(f"- 覆盖算法数: **{n_alg}**")

    # 算法胜场
    if len(wins) > 0:
        lines.append("\n## 解质量胜场（按 UB_mean 最小计）")
        for _, r in wins.sort_values("win_count", ascending=False).head(10).iterrows():
            lines.append(f"- algo={r['algo']}, ablation={r['ablation']}, TL={r['time_limit_sec']}: win_count={int(r['win_count'])}")

    # stop reason
    lines.append("\n## stop_reason 主要分布（每 algo-ablation Top2）")
    tmp = stopdist.copy()
    tmp = tmp.sort_values(["algo","ablation","count"], ascending=[True,True,False])
    for (algo, ab), g in tmp.groupby(["algo","ablation"], dropna=False):
        top = g.head(2)
        items = ", ".join([f"{sr}({share:.2f})" for sr, share in zip(top["stop_reason"], top["share"])])
        lines.append(f"- {algo} | {ab}: {items}")

    # IterLP 是否真正改进：用 best_time_mean 与 UB_mean 的关系写一句“检查点”
    it = summary[summary["algo"].astype(str).str.contains("iterlp", case=False, na=False)]
    if len(it) > 0:
        frac_zero = float((it["best_time_mean_sec"].fillna(0).astype(float) <= 1e-9).mean())
        lines.append("\n## IterLP 改进发生时机（粗略）")
        lines.append(f"- best_time_mean_sec≈0 的比例（表示最好 UB 多半来自初始化）: **{frac_zero:.2f}**")
        lines.append("- 注：该指标需结合具体实例族进一步解读（高密度图更可能出现后期改进）。")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("[Path] INPUT_CSV =", INPUT_CSV)
    print("[Path] OUT_DIR   =", OUT_DIR)
    ensure_dirs()
    df = load_master_runs(INPUT_CSV)

    # 1) 实例概览
    ov = instance_overview_table(df)
    ov.to_csv(OUT_DIR / "data" / "instance_overview.csv", index=False, encoding="utf-8-sig")
    latex_table(
        ov,
        OUT_DIR / "tables" / "instance_overview.tex",
        caption="Instance overview (|V|, |E|, density, avg degree).",
        label="tab:instance_overview",
    )

    # 2) 覆盖度表
    cov = coverage_table(df)
    cov.to_csv(OUT_DIR / "data" / "coverage_instance_algo.csv", index=False, encoding="utf-8-sig")

    # 3) 主汇总（你现在分析的核心）
    summary = agg_runs(df, USE_GROUP)
    summary.to_csv(OUT_DIR / "data" / "summary_long.csv", index=False, encoding="utf-8-sig")

    # 4) 生成“论文友好”的宽表：UB_mean / UB_min
    #    以 instance 为行，algo 为列（对比很直观）
    #    注意：如果你启用 GROUP_FULL，维度会更多，这里建议只对 MIN 版本做宽表
    if USE_GROUP == GROUP_MIN:
        ub_mean_wide = summary.pivot_table(index=["instance","ablation","time_limit_sec"], columns="algo", values="UB_mean")
        ub_min_wide  = summary.pivot_table(index=["instance","ablation","time_limit_sec"], columns="algo", values="UB_min")
        ub_mean_wide.to_csv(OUT_DIR / "data" / "ub_mean_wide.csv", encoding="utf-8-sig")
        ub_min_wide.to_csv(OUT_DIR / "data" / "ub_min_wide.csv", encoding="utf-8-sig")

        latex_table(
            ub_mean_wide.reset_index(),
            OUT_DIR / "tables" / "ub_mean_wide.tex",
            caption="UB_mean comparison (wide table).",
            label="tab:ub_mean_wide",
        )
        latex_table(
            ub_min_wide.reset_index(),
            OUT_DIR / "tables" / "ub_min_wide.tex",
            caption="UB_min comparison (wide table).",
            label="tab:ub_min_wide",
        )

    # 5) 胜场统计
    wins = best_algo_win_count(summary)
    wins.to_csv(OUT_DIR / "data" / "win_count.csv", index=False, encoding="utf-8-sig")
    latex_table(
        wins.sort_values("win_count", ascending=False).head(20),
        OUT_DIR / "tables" / "win_count_top20.tex",
        caption="Top-20 win counts by algorithm (lower UB_mean is better).",
        label="tab:win_count_top20",
    )

    # 6) stop_reason 分布
    stopdist = stop_reason_dist(df)
    stopdist.to_csv(OUT_DIR / "data" / "stop_reason_dist.csv", index=False, encoding="utf-8-sig")
    plot_stop_reason(stopdist)

    # 7) 图：解质量分布、runtime vs n、IterLP anytime
    plot_quality_box(summary)
    plot_runtime_vs_n(summary, df)

    plot_anytime_scatter(df)

    # 8) 自动结论草稿
    write_findings(summary, wins, stopdist, OUT_DIR / "text" / "auto_findings.md")

    print("[OK] Analysis outputs written to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
