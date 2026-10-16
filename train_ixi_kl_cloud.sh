#!/usr/bin/env bash
# Run inside the prepared vqgan environment on a single cloud GPU server.
# Usage:
#   conda activate vqgan
#   DATA_PATH=/data/IXI/3D/IXI_datalist_3D.json bash train_ixi_kl_cloud.sh
# Resume: add RESUME=/path/to/last.ckpt before bash.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
: "${DATA_PATH:?Set DATA_PATH to the cloud IXI_datalist_3D.json path}"
[[ -f "$DATA_PATH" ]] || { echo "Dataset JSON not found: $DATA_PATH" >&2; exit 1; }
EXP_NAME="${EXP_NAME:-ixi_kl}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-kl-ixi}"
export WANDB_NAME="${WANDB_NAME:-${EXP_NAME}}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Keep cloud NCCL defaults so available GPU interconnects can be used.
# Set NCCL_P2P_DISABLE=1 / NCCL_IB_DISABLE=1 externally if required.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DATA_PATH=${DATA_PATH}"
echo "EXP_NAME=${EXP_NAME}"
echo "GPU_IDS=${GPU_IDS:-0,1,}"
nvidia-smi
CMD=(python main.py -b configs/ixi_kl.yaml -t True
  --gpus "${GPU_IDS:-0,1,}" -n "$EXP_NAME"
  "data.params.data_path=${DATA_PATH}"
  "data.params.num_workers=${NUM_WORKERS:-8}"
  "model.params.lossconfig.params.disc_factor=${DISC_FACTOR:-0}")
if [[ -n "${RESUME:-}" ]]; then
  [[ -f "$RESUME" ]] || { echo "Checkpoint not found: $RESUME" >&2; exit 1; }
  CMD+=(--resume_from_checkpoint "$RESUME")
fi
"${CMD[@]}" "$@"
