#!/bin/bash
#SBATCH --job-name=OpenR1_training   # Job name
#SBATCH --gres=gpu:2                 # Request 2 GPUs
#SBATCH --cpus-per-task=16           # Request 16 CPU cores per task
#SBATCH --mem=256G                 # Request 256GB of memory per node
#SBATCH --time=24:00:00              # Maximum runtime (adjust as needed)
#SBATCH --output=training_%j.out     # Output file with job ID

# Load necessary modules
module load cudatoolkit/12.4

# Activate your virtual environment
source openr1/bin/activate

# Run your training script
bash training.sh
