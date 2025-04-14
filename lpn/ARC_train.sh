#!/bin/sh
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=ARC_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/scur2570/lpn/out/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source /home/scur2570/lpn-env/bin/activate
 
cd /home/scur2570/lpn
export WANDB_API_KEY=
export HF_TOKEN =
export PYTHONPATH=${PYTHONPATH}:${PWD}


python src/train.py --config-path configs/arc_train_scaling --config-name 5
