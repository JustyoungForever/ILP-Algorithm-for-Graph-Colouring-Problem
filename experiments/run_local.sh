ROOT="/mnt/g/Nextcloud/FSU_Cloud/毕设/ILP-Algorithm-for-Graph-Colouring-Problem"
DIMACS_DIR="$ROOT/experiments/data/dimacs_selected"
INST="DSJC500.9.col"

# 1) 确认文件存在
ls -lh "$DIMACS_DIR/$INST"

# 2) 做一个“只含该实例”的临时目录（避免 runner 扫到其他图）
ONE_DIR="$ROOT/experiments/data/_one_instance"
mkdir -p "$ONE_DIR"
cp -f "$DIMACS_DIR/$INST" "$ONE_DIR/"

# 3) 跑（输出都在 results/）
mkdir -p "$ROOT/results"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python3 -u "$ROOT/runner.py" --suite dimacs \
  --dimacs-dir "$ONE_DIR" \
  --max-instances 1 \
  --time-limit 600 \
  --algos iterlp2_full \
  --algo-seeds 4 \
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
  --out-csv results_runs.csv \
  --summary-csv results_summary.csv \
  2>&1 | tee "$ROOT/results/run_DSJC1000.9_seed4_$(date +%F_%H%M%S).log"


rm -rf "$ONE_DIR"
