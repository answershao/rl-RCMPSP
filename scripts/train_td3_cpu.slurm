#!/usr/bin/env bash
#SBATCH --job-name=rcmpsp-td3-cpu
#SBATCH --partition=cpu64
#SBATCH --qos=qos_hpc2023098
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/td3_cpu_%j.out
#SBATCH --error=outputs/slurm/td3_cpu_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/slurm

source /hpc/home/hpc2023098/miniconda3/etc/profile.d/conda.sh
conda activate rcmpsp

# Reserve eight CPU threads for the learner and one process for each of the
# 56 environment workers, without nested BLAS thread pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m src.scripts.train_td3 \
  --mode multi \
  --n-envs 56 \
  --total-timesteps 2000000 \
  --buffer-size 200000 \
  --batch-size 256 \
  --device cpu \
  --torch-threads 8 \
  --seed 17 \
  --output-dir outputs/td3_multi_s17_t6400000_b500k_cpu
