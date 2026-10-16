#!/bin/bash
#SBATCH -p asus_pro6000
#SBATCH -q pro6000_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00

set -euo pipefail

source /home/mailab_yonsei/miniconda3/etc/profile.d/conda.sh
conda activate vqgan

PROJECT_DIR="/scratch2/mailab_yonsei/MW/CY/ALDM/VQ-GAN"
STAGE="${STAGE:-2}"   # 1 | 2
EXP_NAME="${EXP_NAME:-adni_vqgan_stage${STAGE}}"
RESUME="${RESUME:-}"                      # logs/<run>/checkpoints/last.ckpt
STAGE1_CKPT="${STAGE1_CKPT:-/scratch2/mailab_yonsei/MW/CY/ALDM/VQ-GAN/logs/2026-07-13T17-45-13_adni_vqgan_stage1/vqgan-adni/2026-07-13T17-45-13_adni_vqgan_stage1/checkpoints/epoch=272-step=99918.ckpt}"            # stage2 only: stage1 checkpoint path
WANDB_PROJECT="${WANDB_PROJECT:-vqgan-adni}"
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

CONFIG="configs/adni_vqgan_stage${STAGE}.yaml"
echo "CONFIG=${CONFIG}"
echo "EXP_NAME=${EXP_NAME}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "GPU_IDS=${GPU_IDS}"
nvidia-smi

CMD=(python main.py -b "${CONFIG}" -t True --gpus "${GPU_IDS}" -n "${EXP_NAME}")

if [[ -n "${RESUME}" ]]; then
  CMD+=(-r "${RESUME}")
fi

# stage2: load stage1 weights via config override
if [[ "${STAGE}" == "2" ]]; then
  if [[ -z "${STAGE1_CKPT}" ]]; then
    echo "ERROR: set STAGE1_CKPT to the stage1 checkpoint before running stage2."
    echo "  e.g. STAGE1_CKPT=logs/<stage1_run>/checkpoints/last.ckpt"
    exit 1
  fi
  CMD+=(model.params.ckpt_path="${STAGE1_CKPT}")
fi

"${CMD[@]}"

# cd /scratch2/mailab_yonsei/MW/CY/ALDM/VQ-GAN && sbatch train_adni.sh