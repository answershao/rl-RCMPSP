#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Environment workers do NumPy/Python scheduling work. Prevent each worker
# from creating its own BLAS thread pool and oversubscribing the host CPUs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTHONUNBUFFERED=1

exec python -m src.scripts.train_ppo \
    --instances-root data/MPLIB2_train_10_500_5 \
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
    --exact-time-limit 60 \
    --exact-workers 1 \
    --output-dir outputs/ppo_gin_a800 \
    "$@"
