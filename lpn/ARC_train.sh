#!/bin/sh
#SBATCH --partition=gpu_a100
#SBATCH --gpus=2
#SBATCH --job-name=ARC_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=/home/scur2570/grad_flow/DL2_Latent/lpn/out/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source /home/scur2570/lpn-env/bin/activate
 
cd /home/scur2570/grad_flow/DL2_Latent/lpn
export WANDB_API_KEY=8e772263ae8e1722c562169fb1c1b602d2d41d2d
export HF_TOKEN=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
export HOME=/gpfs/home5/scur2570
export PYTHONPATH=${PYTHONPATH}:${PWD}


python src/recurrent_train.py --config-name arc_train \
  +training.log_gradients=true \
  +training.log_gradients_every=100 

