module load cudatoolkit/12.4
# Set XDG data (typically used for .local) to a subdirectory of the current working directory
export XDG_DATA_HOME="$(pwd)/.local"

# Set XDG cache directory to a subdirectory of the current working directory
export XDG_CACHE_HOME="$(pwd)/.cache" 

# Set the directory for conda environments
export CONDA_ENVS_PATH="$(pwd)/.conda/envs"

# Set the directory for conda package caches
export CONDA_PKGS_DIRS="$(pwd)/.conda/pkgs"

export HUGGING_FACE_HUB_TOKEN="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"

export TRITON_EXPORT_DIR=$(pwd)/.triton
mkdir -p $(pwd)/.triton
export TRITON_HOME=$(pwd)/.triton


source openr1/bin/activate
mkdir -p .cache .tmp
export XDG_CACHE_HOME=$(pwd)/.cache
export TMPDIR=$(pwd)/.tmp
export CUDA_VISIBLE_DEVICES=0,1,2,3

export TRITON_CACHE_DIR=$(pwd)/.triton

export TRITON_CACHE_DIR=/n/fs/similarity/open-r1
mkdir -p $(pwd)/.cache/huggingface
export XDG_CACHE_HOME=$(pwd)/.cache

mkdir -p $(pwd)/.hf_cache
export HF_HOME=$(pwd)/.hf_cache

mkdir -p $(pwd)/llvm-project
export LLVM_BUILD_DIR=$(pwd)/llvm-project


export TRANSFORMERS_CACHE=$(pwd)/.cache/huggingface/transformers
export HF_DATASETS_CACHE=$(pwd)/.cache/huggingface/datasets

# Train via command line
#accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/grpo.py \
#    --per_device_train_batch_size=1 \
#    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
#    --dataset_name open-r1/OpenR1-Math-220k \
#    --learning_rate 1.0e-5 \
#    --num_train_epochs 5 \
#    --packing \
#    --max_seq_length 8192 \
#    --gradient_checkpointing \
#    --bf16 \
#    --output_dir data/Qwen2.5-7B-Open-R1-Distill-grpo


#nohup CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct  > vllm_serve.log 2>&1 &

CUDA_VISIBLE_DEVICES=1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml \
    --run_name Qwen2.5-1B-GRPO
