#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge many CSV result files in current directory (and subdirs) by schema (column set).
- Fast schema scan (header only)
- Group by column set (ignoring order)
- Write schema_report.csv
- Merge per-schema into merged_schema_<id>.csv (chunked streaming)
- Optionally write one union-merged file with all columns (chunked streaming)

Place this script inside the results/ folder and run:
  python3 merge_results_csvs.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, FrozenSet, Optional

import pandas as pd


# 你“期望的基础列顺序”（用于输出时的 canonical 排序）
BASE_COLUMNS = [
    "instance","family","n","m","density","avg_deg",
    "algo","algo_seed","time_limit_sec",
    "init_heuristic","fix_policy","strong_margin","max_fix_per_round",
    "restarts","perturb_y",
    "LB","UB","gap","feasible","conflicts",
    "runtime_sec","iters","stop_reason",
    "best_time_sec","best_round","ablation",
]

# --------- 配置区（按需改） ----------
OUTPUT_DIRNAME = "_merged"
CHUNKSIZE = 200_000            # 每次读取多少行写出，文件大时可调小
READ_DTYPE_AS_STR = True       # True：更鲁棒（混合类型不炸）；False：保留数值类型
ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "latin1"]  # 遇到编码问题会尝试
WRITE_UNION_FILE = True        # 是否额外生成 union 总表
UNION_FILENAME = "merged_union_all.csv"
# ------------------------------------


@dataclass
class FileSchema:
    path: Path
    columns: List[str]               # 原始顺序（清洗后）
    colset: FrozenSet[str]           # 忽略顺序的集合


def _clean_columns(cols: List[str]) -> List[str]:
    # 去掉 BOM、空格，统一为原样大小写（不强行 lower，避免你区分大小写字段）
    cleaned = []
    seen = set()
    for c in cols:
        if c is None:
            continue
        cc = str(c).replace("\ufeff", "").strip()
        if cc == "":
            continue
        # 避免重复列名导致 pandas 读入时自动加 .1 之类的问题
        if cc in seen:
            # 发现重复列名，先保留，后面会警告并跳过该文件
            cleaned.append(cc)
        else:
            cleaned.append(cc)
            seen.add(cc)
    return cleaned


def _try_read_header_csv(path: Path) -> Optional[List[str]]:
    # 只读第一行 header：速度快、内存小
    for enc in ENCODING_CANDIDATES:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return None
                return _clean_columns(header)
        except UnicodeDecodeError:
            continue
        except Exception:
            # 某些文件可能不是标准 csv 或被占用/损坏
            return None
    return None


def _schema_id_from_colset(colset: FrozenSet[str]) -> str:
    # 用稳定 hash 生成短 id
    joined = "\n".join(sorted(colset))
    h = hashlib.md5(joined.encode("utf-8")).hexdigest()
    return h[:10]


def _canonical_order(cols: List[str]) -> List[str]:
    # 输出时列顺序：先 BASE_COLUMNS 中出现的，再把额外列按字母序附加
    base = [c for c in BASE_COLUMNS if c in cols]
    extra = sorted([c for c in cols if c not in set(BASE_COLUMNS)])
    # 也确保没有遗漏（例如 BASE_COLUMNS 里没写但 schema 里有）
    ordered = base + [c for c in extra if c not in set(base)]
    # 若 schema 有一些奇怪列没进来（理论不会），兜底补齐
    missing = [c for c in cols if c not in set(ordered)]
    return ordered + missing


def scan_schemas(root: Path) -> List[FileSchema]:
    csv_files = sorted([p for p in root.rglob("*.csv") if p.is_file()])

    schemas: List[FileSchema] = []
    for p in csv_files:
        # 跳过输出目录，避免重复合并
        if OUTPUT_DIRNAME in p.parts:
            continue
        header = _try_read_header_csv(p)
        if not header:
            print(f"[WARN] skip (no header or unreadable): {p}", file=sys.stderr)
            continue

        # 检查重复列名
        if len(set(header)) != len(header):
            print(f"[WARN] skip (duplicate column names): {p} | header={header}", file=sys.stderr)
            continue

        schemas.append(FileSchema(path=p, columns=header, colset=frozenset(header)))
    return schemas


def write_schema_report(outdir: Path, groups: Dict[FrozenSet[str], List[FileSchema]]) -> None:
    rows = []
    for colset, files in groups.items():
        sid = _schema_id_from_colset(colset)
        cols_sorted = sorted(list(colset))
        rows.append({
            "schema_id": sid,
            "num_cols": len(colset),
            "num_files": len(files),
            "columns_sorted": "|".join(cols_sorted),
            "example_file_1": str(files[0].path),
            "example_file_2": str(files[1].path) if len(files) > 1 else "",
            "example_file_3": str(files[2].path) if len(files) > 2 else "",
        })

    df = pd.DataFrame(rows).sort_values(["num_cols", "num_files"], ascending=[True, False])
    df.to_csv(outdir / "schema_report.csv", index=False, encoding="utf-8-sig")

    # 额外保存一份 json，方便你后续程序化处理
    schema_map = {
        _schema_id_from_colset(k): {
            "num_cols": len(k),
            "columns_sorted": sorted(list(k)),
            "files": [str(fs.path) for fs in v],
        }
        for k, v in groups.items()
    }
    (outdir / "schema_map.json").write_text(json.dumps(schema_map, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_group_to_csv(out_path: Path, files: List[FileSchema], canonical_cols: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    first = True
    for fs in files:
        # dtype=str 更鲁棒：避免某些文件同一列混合 int/float/str 导致推断不一致
        read_kwargs = dict(low_memory=False)
        if READ_DTYPE_AS_STR:
            read_kwargs["dtype"] = "string"

        try:
            for chunk in pd.read_csv(fs.path, chunksize=CHUNKSIZE, **read_kwargs):
                # 对齐列：缺失补 NA，多余列丢弃（因为该 group 理论上同 schema，但做兜底）
                for c in canonical_cols:
                    if c not in chunk.columns:
                        chunk[c] = pd.NA
                chunk = chunk[canonical_cols]
                chunk.to_csv(out_path, mode="a", index=False, header=first, encoding="utf-8-sig")
                first = False
        except Exception as e:
            print(f"[WARN] merge failed, skip file: {fs.path} | err={e}", file=sys.stderr)
            continue


def merge_union_to_csv(out_path: Path, all_files: List[FileSchema], union_cols: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    first = True
    read_kwargs = dict(low_memory=False)
    if READ_DTYPE_AS_STR:
        read_kwargs["dtype"] = "string"

    for fs in all_files:
        try:
            for chunk in pd.read_csv(fs.path, chunksize=CHUNKSIZE, **read_kwargs):
                for c in union_cols:
                    if c not in chunk.columns:
                        chunk[c] = pd.NA
                chunk = chunk[union_cols]
                # 额外加一列来源文件，方便你追溯（你不需要可以删）
                chunk.insert(0, "__source_file", str(fs.path))
                chunk.to_csv(out_path, mode="a", index=False, header=first, encoding="utf-8-sig")
                first = False
        except Exception as e:
            print(f"[WARN] union merge failed, skip file: {fs.path} | err={e}", file=sys.stderr)
            continue


def main():
    root = Path(__file__).resolve().parent
    outdir = root / OUTPUT_DIRNAME
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Scan] root={root}")
    schemas = scan_schemas(root)
    if not schemas:
        print("[Done] no csv files found or all unreadable.")
        return

    # 按 schema（列集合）分组
    groups: Dict[FrozenSet[str], List[FileSchema]] = {}
    for fs in schemas:
        groups.setdefault(fs.colset, []).append(fs)

    # 写报告
    write_schema_report(outdir, groups)
    print(f"[Report] wrote: {outdir / 'schema_report.csv'}")
    print(f"[Report] wrote: {outdir / 'schema_map.json'}")
    print(f"[Report] schemas={len(groups)} | files={len(schemas)}")

    # 每个 schema 单独合并
    for colset, files in sorted(groups.items(), key=lambda kv: (len(kv[0]), -len(kv[1]))):
        sid = _schema_id_from_colset(colset)
        canonical_cols = _canonical_order(sorted(list(colset)))  # schema 内列做 canonical 排序
        out_path = outdir / f"merged_schema_{sid}_cols{len(colset)}_files{len(files)}.csv"
        print(f"[Merge] schema_id={sid} cols={len(colset)} files={len(files)} -> {out_path.name}")
        merge_group_to_csv(out_path, files, canonical_cols)

    # 可选：生成 union 总表（所有列的并集）
    if WRITE_UNION_FILE:
        union = set()
        for fs in schemas:
            union |= set(fs.columns)
        union_cols = _canonical_order(sorted(list(union)))
        out_path = outdir / UNION_FILENAME
        print(f"[Union] cols={len(union_cols)} files={len(schemas)} -> {out_path.name}")
        merge_union_to_csv(out_path, schemas, union_cols)

    print("[Done]")


if __name__ == "__main__":
    main()
