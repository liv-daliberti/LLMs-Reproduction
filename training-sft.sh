#!/bin/bash
#SBATCH --job-name=OpenR1_SFT
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=128:00:00
#SBATCH --output=training_%j.out

#-------------------------
# Load CUDA module
#-------------------------
module load cudatoolkit/12.4

#-------------------------
# Basic env vars
#-------------------------
export XDG_CACHE_HOME="$(pwd)/.cache"
export CONDA_ENVS_PATH="$(pwd)/.conda/envs"
export CONDA_PKGS_DIRS="$(pwd)/.conda/pkgs"
export HUGGING_FACE_HUB_TOKEN="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"

export TRITON_EXPORT_DIR="$(pwd)/.triton"
mkdir -p "$TRITON_EXPORT_DIR"
export TRITON_HOME="$TRITON_EXPORT_DIR"
export TRITON_CACHE_DIR="/n/fs/similarity/open-r1"

#-------------------------
# Log in to Hugging Face
#-------------------------
echo "$HUGGING_FACE_HUB_TOKEN" | python -m huggingface_hub login --token
echo "Logged in to Hugging Face"

#-------------------------
# Unique run identifiers
#-------------------------
RUN_NAME="Qwen1.5B-SFT-Finetune"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$(pwd)/logs"
TRAINING_LOG="${LOG_DIR}/training_${RUN_NAME}_${TIMESTAMP}.log"

#-------------------------
# Install CLI tools & activate env
#-------------------------
source openr1/bin/activate
pip install --upgrade yq huggingface_hub

#-------------------------
# Create cache & log dirs
#-------------------------
mkdir -p \
  "$(pwd)/.cache/huggingface" \
  "$(pwd)/.hf_cache" \
  "$(pwd)/.tmp" \
  "$(pwd)/llvm-project" \
  "${LOG_DIR}"

export TMPDIR="$(pwd)/.tmp"
export HF_HOME="$(pwd)/.hf_cache"
export LLVM_BUILD_DIR="$(pwd)/llvm-project"
export TRANSFORMERS_CACHE="$(pwd)/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$(pwd)/.cache/huggingface/datasets"

#-------------------------
# Use Slurm’s GPU assignment
#-------------------------
echo "Slurm has set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
ALL_GPUS=$CUDA_VISIBLE_DEVICES
TRAINING_GPUS=$ALL_GPUS
echo "Training will use GPUs: $TRAINING_GPUS"

#-------------------------
# Update Accelerate config
#-------------------------
CONFIG_FILE="recipes/accelerate_configs/zero3.yaml"
cp "${CONFIG_FILE}" "${CONFIG_FILE}.bak"
NUM_TRAINING=$(echo $TRAINING_GPUS | tr ',' '\n' | wc -l)
yq -y --in-place ".num_processes = $NUM_TRAINING" "${CONFIG_FILE}"

#-------------------------
# Source your SFT config
#-------------------------
# this file should export something like SFT_ARGS="--model_name_or_path ... --learning_rate ... etc."
#source /n/fs/similarity/open-r1/recipes/Qwen2.5-1.5B-Instruct/sft/config_demo_liv.yaml

#-------------------------
# Launch distributed SFT training
#-------------------------
export MASTER_PORT=29503
CUDA_VISIBLE_DEVICES=$TRAINING_GPUS \
  ACCELERATE_LOG_LEVEL=info accelerate launch \
    --main_process_port $MASTER_PORT \
    --config_file "${CONFIG_FILE}" \
    src/open_r1/sft.py \
    --config  recipes/Qwen2.5-1.5B-Instruct/sft/config_demo_liv.yaml \
    --run_name "${RUN_NAME}-${TIMESTAMP}" \
  2>&1 | tee "${TRAINING_LOG}"
