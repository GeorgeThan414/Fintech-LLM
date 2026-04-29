#!/bin/bash
#SBATCH --job-name=forecast_batch
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=forecast_%j.out
#SBATCH --error=forecast_%j.err

module load gcc/14.2.0 python/3.14.0 cuda/12.8.1
source ~/pytorch-env/bin/activate

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface/transformers

python3 run_forecast_v2.py >> forecast_${SLURM_JOB_ID}.log 2>&1