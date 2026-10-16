#!/bin/bash
#SBATCH -p gigabyte_a6000,suma_a6000,tyan_a6000
#SBATCH --gres=gpu:2
#SBATCH --exclude=node30
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mwj0528@yonsei.ac.kr

set -euo pipefail

source /home/mailab_yonsei/miniconda3/etc/profile.d/conda.sh
conda activate vqgan

PROJECT_DIR="/scratch2/mailab_yonsei/MW/CREPA_UNet_3D/VQ-GAN"
EXP_NAME="${EXP_NAME:-adni_kl_adv}"
# Set RESUME to a checkpoint path to resume; empty starts a fresh run.
RESUME="${RESUME:-}"
WANDB_PROJECT="${WANDB_PROJECT:-kl-adni}"
WANDB_NAME="${WANDB_NAME:-${EXP_NAME}}"

cd "${PROJECT_DIR}"

# 3D NIfTI + DDP + DataLoader workers can exhaust the default fd limit.
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_NAME="${WANDB_NAME}"

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
GPU_IDS="${GPU_IDS%,},"

CONFIG="configs/adni_kl.yaml"

echo "CONFIG=${CONFIG}"
echo "EXP_NAME=${EXP_NAME}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "GPU_IDS=${GPU_IDS}"
echo "NODE=${SLURMD_NODENAME:-$(hostname)}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

# Initialize every requested GPU before loading the model or checkpoint.
python - "${GPU_IDS}" <<'PY'
import sys
import torch

requested = [int(index) for index in sys.argv[1].split(",") if index]
try:
    available = torch.cuda.device_count()
    if any(index < 0 or index >= available for index in requested):
        raise RuntimeError(
            f"Requested GPU indices {requested}, but CUDA reports {available} GPU(s)."
        )
    for index in requested:
        with torch.cuda.device(index):
            probe = torch.zeros(1, device=f"cuda:{index}")
            torch.cuda.synchronize(index)
        print(f"GPU {index}: {torch.cuda.get_device_name(index)} OK", flush=True)
except Exception as error:
    print(f"GPU preflight failed: {error}", file=sys.stderr)
    print("Check the allocated node/GPU status and resubmit on a healthy node.", file=sys.stderr)
    sys.exit(1)
PY

# -n and -r cannot be used together (main.py).
# Resume: -r only (continues in the same logdir derived from the ckpt path).
# Fresh run: -n for logdir postfix.
CMD=(python main.py -b "${CONFIG}" -t True --gpus "${GPU_IDS}")

if [[ -n "${RESUME}" ]]; then
  echo "RESUME=${RESUME}"
  CMD+=(-r "${RESUME}")
else
  CMD+=(-n "${EXP_NAME}")
fi

"${CMD[@]}"

# cd /scratch2/mailab_yonsei/MW/CREPA_UNet_3D/VQ-GAN && sbatch train_adni_kl.sh
# cd /scratch2/mailab_yonsei/MW/CREPA_UNet_3D/VQ-GAN && RESUME=/scratch2/mailab_yonsei/MW/CREPA_UNet_3D/VQ-GAN/logs/2026-07-21T14-03-26_adni_kl/checkpoints/trainstep_checkpoints/epoch=000068-step=000050000.ckpt sbatch train_adni_kl.sh
