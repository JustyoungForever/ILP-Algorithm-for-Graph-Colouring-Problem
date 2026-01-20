#!/bin/bash
#SBATCH --job-name=final_medium
#SBATCH --partition=s_standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/data/medium/logs/%x_%j.out
#SBATCH --error=experiments/data/medium/logs/%x_%j.err

set -euo pipefail

module load anaconda/2024-02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcp_iterlp

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

conda list > results/env_${SLURM_JOB_ID}.txt
python -V > results/python_${SLURM_JOB_ID}.txt

GRAPH_DIR="experiments/data/medium/graphs"
TL=6000   # 6h

for f in $(ls -1 "${GRAPH_DIR}"/*.col | sort); do
  base=$(basename "${f}" .col)
  tmp="experiments/data/tmp_final_${SLURM_JOB_ID}_${base}"
  mkdir -p "${tmp}"
  cp "${f}" "${tmp}/"

  python -u runner.py --suite dimacs \
    --dimacs-dir "${tmp}" \
    --time-limit "${TL}" \
    --algos dsatur,slo,iterlp2_full \
    --algo-seeds 0,1,2 \
    --restarts 32 \
    --max-fix-per-round 30 \
    --strong-margin 0.25 \
    --perturb-y 1e-6 \
    --save-trace 0 \
    --out-dir results \
    --out-csv "final_medium_${base}_runs_${SLURM_JOB_ID}.csv" \
    --summary-csv "final_medium_${base}_summary_${SLURM_JOB_ID}.csv"

  rm -rf "${tmp}"
done
