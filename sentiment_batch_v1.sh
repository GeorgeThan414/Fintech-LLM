#!/bin/bash
#SBATCH --job-name=sentiment_batch
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=sentiment_%j.out
#SBATCH --error=sentiment_%j.err

module load gcc/14.2.0 python/3.14.0 cuda/12.8.1
source ~/pytorch-env/bin/activate

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface/transformers


export GROQ_API_KEY="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"

python3 run_sentiment.py