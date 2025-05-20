#!/bin/sh
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=cross_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/home/scur2570/DL2_Latent_attention/lpn/out/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source /home/scur2570/lpn-env/bin/activate
 
cd /home/scur2570/DL2_Latent_attention/lpn
export WANDB_API_KEY=8e772263ae8e1722c562169fb1c1b602d2d41d2d
export HF_TOKEN=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
export HOME=/gpfs/home5/scur2570
export PYTHONPATH=${PYTHONPATH}:${PWD}


python src/train.py --config-name cross_test 
