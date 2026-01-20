#!/bin/bash
#SBATCH --job-name=dsjc1000_dsatur
#SBATCH --partition=s_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
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
export GCP_VIZ=0

GRAPH_DIR="$ROOT/experiments/data/large/graphs"
INST="DSJC1000.9.col"
[[ -f "$GRAPH_DIR/$INST" ]] || { echo "[ERROR] Missing $GRAPH_DIR/$INST"; ls -lh "$GRAPH_DIR"; exit 1; }

SEED=0
TL=999999   # DSATUR 不需要 time-limit，给个很大值即可
base="${INST%.col}"

tmp="$ROOT/experiments/data/tmp_${SLURM_JOB_ID}_${base}_s${SEED}"
mkdir -p "$tmp"
ln -f "$GRAPH_DIR/$INST" "$tmp/$INST" 2>/dev/null || cp "$GRAPH_DIR/$INST" "$tmp/"

OUT_RUN="$ROOT/results/time_${base}_dsatur_seed${SEED}_runs_${SLURM_JOB_ID}.csv"
OUT_SUM="$ROOT/results/time_${base}_dsatur_seed${SEED}_summary_${SLURM_JOB_ID}.csv"

echo "[INFO] Start $(date)"
echo "[INFO] Instance=$INST seed=$SEED algo=dsatur"
echo "[INFO] Outputs:"
echo "  $OUT_RUN"
echo "  $OUT_SUM"

START_TS=$(date +%s)

python -u "$ROOT/runner.py" --suite dimacs \
  --dimacs-dir "$tmp" \
  --time-limit "$TL" \
  --algos dsatur \
  --algo-seeds "$SEED" \
  --restarts 1 \
  --max-fix-per-round 1 \
  --save-trace 0 \
  --out-dir "$ROOT/results" \
  --out-csv "$(basename "$OUT_RUN")" \
  --summary-csv "$(basename "$OUT_SUM")"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo "[INFO] End   $(date)"
echo "[INFO] WallClockSec=$ELAPSED"

rm -rf "$tmp"
