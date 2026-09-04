#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

# Environment workers do NumPy/Python scheduling work. Prevent each worker
# from creating its own BLAS thread pool and oversubscribing the host CPUs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTHONUNBUFFERED=1

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TRAIN_LOG_FILE="${LOG_DIR}/train_a800_$(date +%Y%m%d_%H%M%S).log"

nohup python -m scripts.train_ppo \
    --instances-root data/MPLIB2_train_10_50_5 \
    --n-envs 96 \
    --total-timesteps 6400000 \
    --n-steps 256 \
    --batch-size 8192 \
    --n-epochs 2 \
    --gin-layers 2 \
    --device cuda:0 \
    --mixed-precision bf16 \
    --cuda-matmul-precision high \
    --torch-compile \
    --compile-mode reduce-overhead \
    --vec-env subproc \
    --start-method spawn \
    --torch-threads 1 \
    --torch-interop-threads 1 \
    --seed 17 \
    --splits outputs/ppo_gin_a800/splits.json \
    --baseline-results outputs/baselines_mplib2_10_50_5/makespan_summary.csv \
    --output-dir outputs/ppo_gin_a800 \
    "$@" >"${TRAIN_LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "training started in background: PID=${TRAIN_PID}"
echo "log: ${TRAIN_LOG_FILE}"
