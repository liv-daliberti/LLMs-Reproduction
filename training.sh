module load cudatoolkit/12.4
source openr1/bin/activate
mkdir -p .cache .tmp
export XDG_CACHE_HOME=$(pwd)/.cache
export TMPDIR=$(pwd)/.tmp
export CUDA_VISIBLE_DEVICES=0,1

export TRITON_CACHE_DIR=/n/fs/similarity/open-r1
mkdir -p $(pwd)/.cache/huggingface
export XDG_CACHE_HOME=$(pwd)/.cache

mkdir -p $(pwd)/.hf_cache
export HF_HOME=$(pwd)/.hf_cache

#export TRANSFORMERS_CACHE=$(pwd)/.cache/huggingface/transformers
#export HF_DATASETS_CACHE=$(pwd)/.cache/huggingface/datasets

# Train via command line
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --per_device_train_batch_size=1 \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 8192 \
    --gradient_checkpointing \
    --bf16 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
