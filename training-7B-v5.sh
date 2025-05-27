#!/bin/bash
#SBATCH --job-name=OpenR1_GRPO
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=128:00:00
#SBATCH --output=logs/slurm_%j.out

set -e
module load cudatoolkit/12.4
source openr1/bin/activate
pip install --upgrade yq huggingface_hub
# ----------------------------
# Setup
# ----------------------------
export RUN_NAME="Qwen1.5B-GRPO-Finetune"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
#export MODEL_DIR="/n/fs/similarity/open-r1/data/Qwen2.5-7B-Instruct-GRPO-v3"
#export CKPT_DIR="$MODEL_DIR/checkpoint-3500"
export CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_liv.yaml"
export CONFIG_FILE="recipes/accelerate_configs/zero3.yaml"
export SERVER_LOG="logs/liv_vllm_${RUN_NAME}_${TIMESTAMP}.log"
export TRAINING_LOG="logs/liv_train_${RUN_NAME}_${TIMESTAMP}.log"
export HUGGING_FACE_HUB_TOKEN="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"
export TORCH_LOAD_WEIGHTS_ONLY=0

#-------------------------
# Log in to Hugging Face
#-------------------------
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"
echo "✅ Logged into Hugging Face"



# WandB cache and artifact dirs on /n/fs
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
export VLLM_USAGE_STATS_PATH=/n/fs/similarity/vllm/usage_stats.json
export TMPDIR=/n/fs/similarity/wandb-offload/tmp

mkdir -p /n/fs/similarity/vllm

mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

# Optional: Set WANDB_CONFIG_DIR if needed (e.g. wandb/settings)
export WANDB_CONFIG_DIR=/n/fs/similarity/wandb-offload/config

mkdir -p /n/fs/similarity/wandb-offload/{tmp,artifacts,cache,config}
mkdir -p logs .cache .hf_cache .tmp .torchinductor .triton

# HF + Cache
export TRANSFORMERS_CACHE=$(pwd)/.cache/huggingface/transformers
export HF_HOME=$(pwd)/.hf_cache
export HF_DATASETS_CACHE=$(pwd)/.cache/huggingface/datasets
export XDG_CACHE_HOME=$(pwd)/.cache
export TMPDIR=$(pwd)/.tmp
export VLLM_API_KEY="dummy"
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torchinductor
export TRITON_CACHE_DIR=$(pwd)/.triton
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/n/fs/similarity/open-r1/.hf_cache
export XDG_CACHE_HOME="$(pwd)/.cache"
export TRANSFORMERS_CACHE="$(pwd)/.cache/huggingface/transformers"
export HF_HOME="$(pwd)/.hf_cache"
export HF_DATASETS_CACHE="$(pwd)/.cache/huggingface/datasets"
export WANDB_DIR="$(pwd)/.wandb"                           # wandb metadata
export WANDB_CACHE_DIR="$(pwd)/.wandb_cache"              # wandb artifact staging
export TMPDIR="$(pwd)/.tmp"                               # tempfile use
mkdir -p "$XDG_CACHE_HOME" "$TRANSFORMERS_CACHE" "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

# ✅ Force full state loading in PyTorch (not just weights)
export TORCH_LOAD_WEIGHTS_ONLY=0

# (Optional) prevent Triton cache slowdown warnings
#export TRITON_CACHE_DIR="/tmp/$USER/triton"
#mkdir -p "$TRITON_CACHE_DIR"
export TRITON_CACHE_DIR="$(pwd)/.triton"
mkdir -p "$TRITON_CACHE_DIR"

# W&B Online Mode
export WANDB_MODE=online
#export WANDB_PROJECT=your_project_name
#export WANDB_ENTITY=your_entity
#export WANDB_API_KEY=your_token_here  # or ensure ~/.netrc has token

# -----------------------------------
# 1) Launch vLLM server on GPU 0
# -----------------------------------
export VLLM_ATTENTION_BACKEND=xformers
#export VLLM_ENGINE=v0
#unset VLLM_ATTENTION_BACKEND

# Single srun context
srun --gres=gpu:8 --cpus-per-task=64 bash -c "
  # Step 1: vLLM on GPU 0
  export CUDA_VISIBLE_DEVICES=0
  echo 'Launching vLLM on GPU 0...'
  trl vllm-serve \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dtype float16 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    > '$SERVER_LOG' 2>&1 &

  VLLM_PID=\$!

  # Step 2: Health check loop
  until curl -sf http://localhost:8000/health > /dev/null; do
    echo 'Waiting for vLLM...'
    sleep 2
  done
  echo '✅ vLLM is healthy'

  # Step 3: Training on GPUs 1–7
  export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
  echo '🚀 Launching training on GPUs 1–7...'
  accelerate launch \
    --main_process_port 29504 \
    --config_file '$CONFIG_FILE' \
    src/open_r1/grpo.py \
    --config '$CONFIG' \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --run_name '${RUN_NAME}-${TIMESTAMP}' \
    --ignore_data_skip \
    --seed 42 \
    > '$TRAINING_LOG' 2>&1

  wait \$VLLM_PID
"
