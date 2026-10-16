#!/bin/bash
#SBATCH -p gigabyte_pro6000,asus_pro6000
#SBATCH -q pro6000_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mwj0528@yonsei.ac.kr

set -euo pipefail

source /home/mailab_yonsei/miniconda3/etc/profile.d/conda.sh
conda activate vqgan

PROJECT_DIR="/scratch2/mailab_yonsei/MW/CREPA_UNet_3D/VQ-GAN"
EXP_NAME="${EXP_NAME:-brats_kl}"
# Set RESUME= to start a fresh run. Default resumes the latest stage-1 KL ckpt.
RESUME="${RESUME:-}"
WANDB_PROJECT="${WANDB_PROJECT:-kl-brats}"
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
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((NUM_GPUS - 1)))},"

CONFIG="configs/brats_kl.yaml"

echo "CONFIG=${CONFIG}"
echo "EXP_NAME=${EXP_NAME}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "GPU_IDS=${GPU_IDS}"
nvidia-smi

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

# cd /scratch2/mailab_yonsei/MW/CREPA_UNet_3D/VQ-GAN && sbatch train_brats_kl.sh
# Fresh: RESUME= sbatch train_brats_kl.sh
