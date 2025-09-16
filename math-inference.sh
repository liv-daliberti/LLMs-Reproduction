#!/usr/bin/env bash
#SBATCH --job-name=infer_math_ckpts_local
#SBATCH --output=logs/infer_ckpts_local_%A_%a.out
#SBATCH --error=logs/infer_ckpts_local_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
#SBATCH --array=0-30
set -euo pipefail
ulimit -n 4096

export LOGLEVEL=DEBUG
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

#module load cudatoolkit/12.4
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh

# ── Env ─────────────────────────────────────────────────────
export ROOT_DIR="$PWD"
export ENV_NAME="openr1"
export ENV_DIR="$ROOT_DIR/$ENV_NAME"
conda activate "$ENV_DIR"
python --version

conda activate "$ENV_DIR"
echo "✅ Conda env active at: $(which python)"
python --version

# ---- ensure we don't pull packages from ~/.local ----
export PYTHONNOUSERSITE=1    # ignore user site-packages
unset PYTHONPATH             # avoid stray paths shadowing the env
export PIP_DISABLE_PIP_VERSION_CHECK=1

# (You can leave Hub online for dataset downloads)
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_REQUEST_TIMEOUT=60

# Caches
export WANDB_MODE=online
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
export TMPDIR="$(pwd)/.tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="$(pwd)/.torchinductor"
export TRITON_CACHE_DIR="$(pwd)/.triton"

mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" "$TMPDIR" logs .hf_cache .triton .torchinductor

# ── Local model root (has tokenizer.json + checkpoint-* dirs) ───────────
# Set this to your actual path; if you're already cd'ed into it, use "$PWD".
MODEL_ROOT="/n/fs/similarity/open-r1/data/Qwen2.5-1.5B-Open-R1-GRPO-math-v1" # e.g., /n/fs/similarity/.../Qwen2.5-1.5B-Open-R1-GRPO-math-v1

# Enumerate the exact local checkpoints you showed:
CHECKPOINT_STEPS=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550)

STEP=${CHECKPOINT_STEPS[$SLURM_ARRAY_TASK_ID]}
CKPT_DIR="$MODEL_ROOT/checkpoint-$STEP"

echo "→ Using local checkpoint: $CKPT_DIR"
test -d "$CKPT_DIR" || { echo "Missing $CKPT_DIR"; exit 1; }

# ── Outputs ─────────────────────────────────────────────────
OUTPUT_ROOT="/n/fs/similarity/open-r1/results/GRPO-1.5B"
OUTDIR="$OUTPUT_ROOT/step${STEP}"
CACHEROOT="$OUTPUT_ROOT/hf_cache"
mkdir -p "$OUTDIR" "$CACHEROOT"

export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"

# Your inference script path
SCRIPT_PATH="/n/fs/similarity/open-r1/math-inference.py"

# ── Run ─────────────────────────────────────────────────────
python -u "$SCRIPT_PATH" \
  --model_name_or_path "$CKPT_DIR" \
  --output_dir "$OUTDIR" \
  --batch_size 1 \
  --entropy_mode full \
  --num_examples 1000 \
  --num_samples 8 \
  --temperature 0.7 \
  --top_p 0.95 \
  --seed 42 \
  --dtype bfloat16 \
  --dataset_id MATH-500 \
  --split test \
  --two_pass \
  --second_pass_phrase "Wait, we need to reconsider. Let's think this through step by step." \
  --think_cap 750 \
  --answer_cap 50 \
  --step "$STEP"
  
echo "✓ Step $STEP complete → $OUTDIR"
