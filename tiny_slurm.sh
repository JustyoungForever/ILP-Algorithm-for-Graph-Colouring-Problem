#!/bin/bash
#SBATCH --job-name=iterlp_tiny
#SBATCH --partition=b_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=experiments/data/tiny/logs/%x_%j.out
#SBATCH --error=experiments/data/tiny/logs/%x_%j.err

set -euo pipefail


# ====== Conda 环境 ======
module load anaconda/2024-02-1
conda activate gcp_iterlp

# ====== 线程控制 ======
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ====== tiny 跑通配置：不追最优，只确保可行+不报错 ======
python -u runner.py --suite dimacs \
  --dimacs-dir experiments/data/tiny/graphs \
  --time-limit 5 \
  --algos dsatur,slo,iterlp2_full \
  --algo-seeds 0 \
  --restarts 1 \
  --max-fix-per-round 5 \
  --strong-margin 0.25 \
  --perturb-y 1e-6 \
  --save-trace 0 \
  --out-dir results \
  --out-csv tiny_runs_${SLURM_JOB_ID}.csv \
  --summary-csv tiny_summary_${SLURM_JOB_ID}.csv
