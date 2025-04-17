#!/bin/sh
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=ARC_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/scur2570/lpn/out/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source /home/scur2570/lpn-env/bin/activate
 
cd /home/scur2570/lpn
export WANDB_API_KEY=8e772263ae8e1722c562169fb1c1b602d2d41d2d
export HF_TOKEN = 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
export PYTHONPATH=${PYTHONPATH}:${PWD}

python src/evaluate_checkpoint.py \
  -w alisia-baielli/ARC/fiery-dawn-4--checkpoint:latest \
  -jc json/arc-agi_training_challenges.json \
  -js json/arc-agi_training_solutions.json \
  -i gradient_ascent \
  --num-steps 100 \
  --lr 1.0 \
  --optimizer adam
