#!/bin/bash
#SBATCH -p asus_pro6000
#SBATCH -q pro6000_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

set -euo pipefail

source /home/mailab_yonsei/miniconda3/etc/profile.d/conda.sh
conda activate vqgan

PROJECT_DIR="/scratch2/mailab_yonsei/MW/CY/ALDM/VQ-GAN"
STAGE="${STAGE:-2}"
CONFIG="${CONFIG:-configs/adni_vqgan_stage${STAGE}.yaml}"
SPLIT="${SPLIT:-test}"
DATA_RANGE="${DATA_RANGE:-1.0}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
EXP_NAME="${EXP_NAME:-adni_vqgan_stage${STAGE}}"
RUN_NAME="${RUN_NAME:-}"
LOGDIR="${LOGDIR:-}"
CKPT="${CKPT:-/scratch2/mailab_yonsei/MW/CY/ALDM/VQ-GAN/logs/2026-07-14T16-34-57_adni_vqgan_stage2/vqgan-adni/2026-07-14T16-34-57_adni_vqgan_stage2/checkpoints/epoch=452-step=165798.ckpt}"
OUTDIR="${OUTDIR:-${PROJECT_DIR}/results/adni_recon_${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
MODALITIES=(t1 t2 pd)

cd "${PROJECT_DIR}"

ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [[ -z "${LOGDIR}" ]]; then
  if [[ -n "${RUN_NAME}" ]]; then
    LOGDIR="${PROJECT_DIR}/logs/${RUN_NAME}"
  else
    LOGDIR="$(ls -td "${PROJECT_DIR}/logs/"*_"${EXP_NAME}" 2>/dev/null | head -n 1 || true)"
  fi
fi

if [[ -z "${CKPT}" && -n "${LOGDIR}" && -d "${LOGDIR}" ]]; then
  CKPT="$(find "${LOGDIR}" -name '*.ckpt' -type f 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found."
  echo "  Set CKPT explicitly, or set LOGDIR/EXP_NAME to a valid training run."
  echo "  e.g. CKPT=logs/.../checkpoints/epoch=208-step=76494.ckpt"
  exit 1
fi

mkdir -p "${OUTDIR}"

echo "CONFIG=${CONFIG}"
echo "LOGDIR=${LOGDIR:-<not set>}"
echo "CKPT=${CKPT}"
echo "OUTDIR=${OUTDIR}"
echo "SPLIT=${SPLIT}"
echo "MODALITIES=${MODALITIES[*]}"
nvidia-smi || echo "WARN: nvidia-smi unavailable (continuing)"

python taming/scripts/reconstruct.py \
  -b "${CONFIG}" \
  -r "${CKPT}" \
  --outdir "${OUTDIR}" \
  --modalities "${MODALITIES[@]}" \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --data_range "${DATA_RANGE}" \
  --gpu

echo ""
echo "Done. Results in ${OUTDIR}"

# cd /scratch2/mailab_yonsei/MW/CY/ALDM/VQ-GAN && sbatch test_adni.sh