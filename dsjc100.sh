#!/bin/bash
#SBATCH --job-name=dsjc500_iterlp_live_8h
#SBATCH --partition=b_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=08:20:00
#SBATCH --signal=B:TERM@1200
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
export GCP_VIZ=0

GRAPH_DIR="$ROOT/experiments/data/large/graphs"
INST="DSJC500.9.col"
[[ -f "$GRAPH_DIR/$INST" ]] || { echo "[ERROR] Missing $GRAPH_DIR/$INST"; ls -lh "$GRAPH_DIR"; exit 1; }

SEED=0

# ---- 20分钟算法预算 + buffer ----
TL=28800               # 10 minutes algorithm budget
SOFT_TL=$((TL-120))     # leave 60s for graceful shutdown (runner flushes every --live-every)

base="${INST%.col}"

tmp="$ROOT/experiments/data/tmp_${SLURM_JOB_ID}_${base}_s${SEED}"
mkdir -p "$tmp"
ln -f "$GRAPH_DIR/$INST" "$tmp/$INST" 2>/dev/null || cp "$GRAPH_DIR/$INST" "$tmp/"

OUT_RUN="$ROOT/results/time_${base}_iterlp_seed${SEED}_runs_${SLURM_JOB_ID}.csv"
OUT_SUM="$ROOT/results/time_${base}_iterlp_seed${SEED}_summary_${SLURM_JOB_ID}.csv"

echo "[INFO] Start $(date)"
echo "[INFO] Instance=$INST seed=$SEED algo=iterlp2_full TL=${TL}s (soft=${SOFT_TL}s)"
echo "[INFO] Outputs:"
echo "  $OUT_RUN"
echo "  $OUT_SUM"

START_TS=$(date +%s)

# Pre-create files so you can see them immediately even if killed very early
touch "$OUT_RUN" "$OUT_SUM" || true

PY_PID=""
on_term() {
  echo "[WARN] Caught SIGTERM/SIGINT at $(date). Forwarding SIGINT to python for graceful shutdown..."
  if [[ -n "${PY_PID:-}" ]]; then
    kill -INT "$PY_PID" 2>/dev/null || true
    # give it a brief chance to flush (do not hang forever)
    timeout 20s wait "$PY_PID" 2>/dev/null || true
  fi
  echo "[WARN] CSV status after interrupt:"
  ls -lh "$OUT_RUN" "$OUT_SUM" 2>/dev/null || true
  echo "[WARN] Tail runs.csv (if exists):"
  tail -n 5 "$OUT_RUN" 2>/dev/null || true
}
trap on_term TERM INT

# Don't let non-zero exit code prevent post-run checks & sync
set +e

echo "[INFO] runner path: $ROOT/runner.py"
python -u "$ROOT/runner.py" -h 2>&1 | grep -q -- "--live-update" || {
  echo "[ERROR] runner.py does NOT support --live-update (wrong version on cluster)."
  python -u "$ROOT/runner.py" -h | head -n 80
  exit 2
}
md5sum "$ROOT/runner.py" | sed 's/^/[INFO] runner md5: /'

python -u "$ROOT/runner.py" --suite dimacs \
  --dimacs-dir "$tmp" \
  --time-limit "$SOFT_TL" \
  --algos iterlp2_full \
  --algo-seeds "$SEED" \
  --restarts 32 \
  --max-fix-per-round 10 \
  --edge-mode lazy \
  --lazy-threshold 20000000 \
  --strong-margin 0.25 \
  --perturb-y 1e-6 \
  --save-trace 0 \
  --live-update 1 \
  --live-every 30 \
  --out-dir "$ROOT/results" \
  --out-csv "$(basename "$OUT_RUN")" \
  --summary-csv "$(basename "$OUT_SUM")" &
PY_PID=$!

wait "$PY_PID"
RC=$?
set -e

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo "[INFO] End   $(date)"
echo "[INFO] WallClockSec=$ELAPSED"
echo "[INFO] runner.py exit code=$RC"

# Force flush to disk to maximize chance CSV is persisted
sync
ls -lh "$OUT_RUN" "$OUT_SUM" 2>/dev/null || true
echo "[INFO] Tail runs.csv:"
tail -n 5 "$OUT_RUN" 2>/dev/null || true
echo "[INFO] Tail summary.csv:"
tail -n 5 "$OUT_SUM" 2>/dev/null || true

rm -rf "$tmp"
