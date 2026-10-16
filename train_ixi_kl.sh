#!/bin/bash
#SBATCH -p gigabyte_a6000,asus_6000ada,suma_a6000,tyan_a6000
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mwj0528@yonsei.ac.kr

set -euo pipefail

source /home/mailab_yonsei/miniconda3/etc/profile.d/conda.sh
conda activate vqgan

PROJECT_DIR="/scratch2/mailab_yonsei/MW/CREPA_UNet_3D/VQ-GAN"
EXP_NAME="${EXP_NAME:-ixi_kl}"
# Set RESUME to a checkpoint to resume in a new log directory; empty starts fresh.
RESUME="${RESUME:-}"
DISC_FACTOR="${DISC_FACTOR:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-kl-ixi}"
WANDB_NAME="${WANDB_NAME:-${EXP_NAME}}"

cd "${PROJECT_DIR}"

# 3D NIfTI + DDP + DataLoader workers can exhaust the default fd limit.
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_NAME="${WANDB_NAME}"

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((NUM_GPUS - 1)))},"

CONFIG="configs/ixi_kl.yaml"

echo "CONFIG=${CONFIG}"
echo "EXP_NAME=${EXP_NAME}"
echo "DISC_FACTOR=${DISC_FACTOR}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "GPU_IDS=${GPU_IDS}"
nvidia-smi

# Create logs/<timestamp>_<EXP_NAME> for both fresh and resumed runs.
CMD=(python main.py -b "${CONFIG}" -t True --gpus "${GPU_IDS}" -n "${EXP_NAME}"
  "model.params.lossconfig.params.disc_factor=${DISC_FACTOR}")

if [[ -n "${RESUME}" ]]; then
  echo "RESUME=${RESUME}"
  if [[ ! -f "${RESUME}" ]]; then
    echo "Checkpoint not found: ${RESUME}" >&2
    exit 1
  fi
  CMD+=(--resume_from_checkpoint "${RESUME}")
fi

"${CMD[@]}"

# cd /scratch2/mailab_yonsei/MW/CREPA_UNet_3D/VQ-GAN && sbatch train_ixi_kl.sh

# RESUME=/scratch2/mailab_yonsei/MW/CREPA_UNet_3D/VQ-GAN/logs/2026-10-12T10-50-25_ixi_kl/checkpoints/last.ckpt sbatch train_ixi_kl.sh
