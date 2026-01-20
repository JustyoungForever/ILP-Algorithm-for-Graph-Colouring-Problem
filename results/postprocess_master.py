#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

BASE_COLUMNS = [
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
    "LB","UB","gap","conflicts","runtime_sec","iters","best_time_sec","best_round",
]

def to_bool(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().lower()
    if s in ("true","1","yes","y","t"):
        return True
    if s in ("false","0","no","n","f"):
        return False
    return pd.NA

def main():
    root = Path(__file__).resolve().parent
    merged_dir = root / "_merged"
    if not merged_dir.exists():
        raise SystemExit(f"cannot find {merged_dir}, please run merge script first.")

    # 找 cols26 / cols25 的 merged 文件（不依赖 schema_id）
    f26 = sorted(merged_dir.glob("merged_schema_*_cols26_*.csv"))
    f25 = sorted(merged_dir.glob("merged_schema_*_cols25_*.csv"))

    if not f26 and not f25:
        raise SystemExit("no merged_schema_* files found in _merged/")

    dfs = []

    def read_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, dtype="string", low_memory=False)
        df["__source_file"] = str(path)
        return df

    for f in f26:
        dfs.append(read_csv(f))
    for f in f25:
        df = read_csv(f)
        if "ablation" not in df.columns:
            df["ablation"] = "baseline"
        else:
            df["ablation"] = df["ablation"].fillna("baseline")
        dfs.append(df)

    runs = pd.concat(dfs, ignore_index=True)

    # 补齐缺列（防御性）
    for c in BASE_COLUMNS:
        if c not in runs.columns:
            runs[c] = pd.NA

    # 统一 feasible 类型
    runs["feasible"] = runs["feasible"].apply(to_bool)

    # 数值列统一为 numeric
    for c in NUMERIC_COLS:
        runs[c] = pd.to_numeric(runs[c], errors="coerce")

    # ablation 缺失：baseline 默认标记（你也可以用 "unknown"）
    runs["ablation"] = runs["ablation"].fillna("baseline")

    # 输出 master_runs
    out_master = merged_dir / "master_runs.csv"
    runs[["__source_file"] + BASE_COLUMNS].to_csv(out_master, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {out_master}")

    # 重复检查（不强制去重，只报告）
    key_cols = [
        "instance","algo","algo_seed","time_limit_sec",
        "init_heuristic","fix_policy","strong_margin","max_fix_per_round",
        "restarts","perturb_y","ablation",
    ]
    run_key = runs[key_cols].astype("string").fillna("").agg("|".join, axis=1)
    dup = runs.loc[run_key.duplicated(keep=False), ["__source_file"] + key_cols + ["UB","runtime_sec","stop_reason"]]
    dup_out = merged_dir / "duplicate_report.csv"
    dup.to_csv(dup_out, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {dup_out} (rows={len(dup)})")

    # 生成 summary（推荐标准分组：包含参数）
    group_cols = [
        "instance","family","n","m","density","avg_deg",
        "algo","ablation","time_limit_sec",
        "init_heuristic","fix_policy","strong_margin","max_fix_per_round",
        "restarts","perturb_y",
    ]

    def feasible_rate(x):
        x = x.dropna()
        if len(x) == 0:
            return pd.NA
        return float((x == True).mean())

    summary = runs.groupby(group_cols, dropna=False).agg(
        runs=("UB","size"),
        feasible_rate=("feasible", feasible_rate),
        UB_min=("UB","min"),
        UB_mean=("UB","mean"),
        UB_std=("UB","std"),
        runtime_mean_sec=("runtime_sec","mean"),
        best_time_mean_sec=("best_time_sec","mean"),
    ).reset_index()

    out_summary = merged_dir / "master_summary.csv"
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {out_summary}")

    # stop_reason 分布表（可直接用于论文文字）
    stop = (runs.groupby(["algo","ablation","stop_reason"], dropna=False)
                .size().reset_index(name="count")
                .sort_values(["algo","ablation","count"], ascending=[True,True,False]))
    out_stop = merged_dir / "stop_reason_summary.csv"
    stop.to_csv(out_stop, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {out_stop}")

if __name__ == "__main__":
    main()
