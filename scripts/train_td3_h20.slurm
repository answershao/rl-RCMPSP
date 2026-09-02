#!/usr/bin/env bash
#SBATCH --job-name=rcmpsp-td3
#SBATCH --partition=sri-gpu
#SBATCH --qos=qos_hejianbo
#SBATCH --gres=gpu:H20:1
#SBATCH --cpus-per-task=80
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=outputs/slurm/td3_%j.out
#SBATCH --error=outputs/slurm/td3_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/slurm

source /hpc/home/hpc2023098/miniconda3/etc/profile.d/conda.sh
conda activate rcmpsp

# Keep the learner within its allocation; each SubprocVecEnv worker is a
# separate process and should not create an OpenMP thread pool of its own.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA PyTorch is required for this H20 training job")
print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
PY

python -m src.scripts.train_td3 \
  --mode multi \
  --n-envs 64 \
  --total-timesteps 2000000 \
  --buffer-size 200000 \
  --batch-size 256 \
  --device cuda \
  --torch-threads 8 \
  --seed 17 \
  --output-dir outputs/td3_multi_s17_t6400000_b500k
