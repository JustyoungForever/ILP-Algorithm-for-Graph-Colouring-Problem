#!/bin/bash
#SBATCH --job-name=dsjc1000_smoke
#SBATCH --partition=s_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/ta32xoy/masterarbeit
#SBATCH --output=/home/ta32xoy/masterarbeit/experiments/data/large/logs/%x_%j.out
#SBATCH --error=/home/ta32xoy/masterarbeit/experiments/data/large/logs/%x_%j.err

set -euo pipefail

ROOT="/home/ta32xoy/masterarbeit"
cd "$ROOT"
mkdir -p "$ROOT/experiments/data/large/logs" "$ROOT/results"

module load anaconda/2024-02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcp_iterlp

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

GRAPH_DIR="$ROOT/experiments/data/large/graphs"
INST="DSJC1000.9.col"
[[ -f "$GRAPH_DIR/$INST" ]] || { echo "[ERROR] Missing $GRAPH_DIR/$INST"; ls -lh "$GRAPH_DIR"; exit 1; }

TL=60
GRACE=300                 # 关键：给足写文件/清理的时间
HARD=$((TL + GRACE))
SEED=0

base="${INST%.col}"
OUT_RUN="$ROOT/results/smoke_${base}_seed${SEED}_runs_${SLURM_JOB_ID}.csv"
OUT_SUM="$ROOT/results/smoke_${base}_seed${SEED}_summary_${SLURM_JOB_ID}.csv"

tmp="$ROOT/experiments/data/tmp_${SLURM_JOB_ID}_${base}_s${SEED}"
mkdir -p "$tmp"
ln -f "$GRAPH_DIR/$INST" "$tmp/$INST" 2>/dev/null || cp "$GRAPH_DIR/$INST" "$tmp/"

echo "[INFO] Running $INST seed=$SEED TL=${TL}s HARD=${HARD}s"
echo "[INFO] Expect outputs:"
echo "  $OUT_RUN"
echo "  $OUT_SUM"
date

python -u "$ROOT/runner.py" --suite dimacs \
  --dimacs-dir "$tmp" \
  --time-limit "$TL" \
  --algos dsatur,iterlp2_full \
  --algo-seeds "$SEED" \
  --restarts 1 \
  --max-fix-per-round 1 \
  --strong-margin 0.25 \
  --perturb-y 1e-6 \
  --save-trace 0 \
  --out-dir "$ROOT/results" \
  --out-csv "$(basename "$OUT_RUN")" \
  --summary-csv "$(basename "$OUT_SUM")" \
  &
PID=$!

# watchdog：HARD 秒后 INT，再等 120 秒才 KILL（非常保守）
KILL_GRACE=120
(
  sleep "$HARD"
  echo "[WATCHDOG] HARD limit ${HARD}s reached, sending SIGINT to PID=$PID"
  kill -INT "$PID" 2>/dev/null || exit 0
  sleep "$KILL_GRACE"
  echo "[WATCHDOG] still running after ${KILL_GRACE}s, sending SIGKILL to PID=$PID"
  kill -KILL "$PID" 2>/dev/null || true
) &
WATCHDOG=$!

wait "$PID"
RC=$?
kill "$WATCHDOG" 2>/dev/null || true

rm -rf "$tmp"

echo "[INFO] runner exit code: $RC"
date

# 关键：没有输出文件就当失败（避免“COMPLETED但啥也没有”）
if [[ ! -s "$OUT_SUM" || ! -s "$OUT_RUN" ]]; then
  echo "[ERROR] Output CSV missing or empty."
  echo "[ERROR] runs exists?  $(test -s "$OUT_RUN" && echo yes || echo no)"
  echo "[ERROR] sum  exists?  $(test -s "$OUT_SUM" && echo yes || echo no)"
  exit 2
fi

# runner 非0也判失败
if [[ $RC -ne 0 ]]; then
  echo "[ERROR] runner failed with code $RC"
  exit $RC
fi

echo "[INFO] Success: outputs written."
