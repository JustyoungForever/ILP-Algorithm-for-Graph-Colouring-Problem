#!/bin/bash
#SBATCH --job-name=dsjc_pool_onesrun
#SBATCH --partition=s_standard
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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

# 三个实例
INSTS=("DSJC250.9.col" "DSJC500.9.col" "DSJC1000.9.col")
# 冒烟：每个seed 3秒；正式实验把它换成你要的 (7200 21600 46800) 等
TLS=(3 3 3)
# 5个seed
SEEDS=(0 1)

# 一次性启动 15 个 task（0..14），避免创建很多 job step
srun -N 1 -n 15 -c 1 --cpu-bind=cores bash -lc '
  set -euo pipefail
  ROOT="/home/ta32xoy/masterarbeit"
  GRAPH_DIR="$ROOT/experiments/data/large/graphs"

  INSTS=("DSJC250.9.col" "DSJC500.9.col" "DSJC1000.9.col")
  TLS=(3 3 3)
  SEEDS=(0 1 2 3 4)

  tid=$SLURM_PROCID
  inst_idx=$((tid / 5))
  seed_idx=$((tid % 5))

  inst=${INSTS[$inst_idx]}
  tl=${TLS[$inst_idx]}
  seed=${SEEDS[$seed_idx]}
  base=${inst%.col}

  # 每个task独立临时目录，避免互相覆盖
  tmp="$ROOT/experiments/data/tmp_${SLURM_JOB_ID}_${base}_s${seed}"
  mkdir -p "$tmp"
  cp "$GRAPH_DIR/$inst" "$tmp/"

  echo "[TASK $tid] inst=$inst seed=$seed TL=${tl}s"

  python -u "$ROOT/runner.py" --suite dimacs \
    --dimacs-dir "$tmp" \
    --time-limit "$tl" \
    --algos dsatur,slo,iterlp2_full \
    --algo-seeds "$seed" \
    --restarts 2 \
    --max-fix-per-round 2 \
    --strong-margin 0.25 \
    --perturb-y 1e-6 \
    --save-trace 0 \
    --out-dir "$ROOT/results" \
    --out-csv "pool_${base}_seed${seed}_runs_${SLURM_JOB_ID}.csv" \
    --summary-csv "pool_${base}_seed${seed}_summary_${SLURM_JOB_ID}.csv"

  rm -rf "$tmp"
'
