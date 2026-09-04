#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

# This host has 52 physical cores across two NUMA nodes. Keep NumPy work in
# each environment single-threaded and reserve CPU capacity for PPO updates.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export PYTHONUNBUFFERED=1

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TRAIN_LOG_FILE="${LOG_DIR}/train_cpu_$(date +%Y%m%d_%H%M%S).log"

# Keep the total CPU footprint close to the 52 physical cores on this host.
# These can be overridden for benchmarking, for example N_ENVS=40.
N_ENVS="${N_ENVS:-32}"
N_STEPS="${N_STEPS:-384}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
TORCH_THREADS="${TORCH_THREADS:-20}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"
MAKESPAN_MIN_DELTA="${MAKESPAN_MIN_DELTA:-0}"

nohup python -m scripts.train_ppo \
    --instances-root data/MPLIB2_train_10_50_5 \
    --n-envs "${N_ENVS}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --n-steps "${N_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --n-epochs 2 \
    --gin-layers 2 \
    --device cpu \
    --mixed-precision none \
    --vec-env subproc \
    --start-method spawn \
    --torch-threads "${TORCH_THREADS}" \
    --torch-interop-threads 1 \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --makespan-min-delta "${MAKESPAN_MIN_DELTA}" \
    --seed 17 \
    --splits outputs/ppo_gin_cpu/splits.json \
    --baseline-results outputs/baselines_mplib2_10_50_5/makespan_summary.csv \
    --output-dir outputs/ppo_gin_cpu \
    "$@" >"${TRAIN_LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "training started in background: PID=${TRAIN_PID}"
echo "log: ${TRAIN_LOG_FILE}"
