#!/bin/bash

#SBATCH --account=uerdogan
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name=exptamed_v0_betatest  # Job name



#SBATCH --partition=akya-cuda
#SBATCH -N 3                       # Request 4 nodes (1 node per optimizer)
#SBATCH --ntasks=3                 # 4 tasks total (1 per node)
#SBATCH --cpus-per-task=10         # 10 CPU cores per task
#SBATCH --gres=gpu:1               # 1 GPU per node
#SBATCH --time=00:45:00             # Time limit per task
#SBATCH --output=optimizer_%j_%t.out  # Output file for each task

# Load necessary modules

module load apps/truba-ai/gpu-2024.0

conda activate gpu-2024.0

# Define optimizer tasks
srun --ntasks=1 --nodes=1 --exclusive python  main.py --dataset cifar10 --total_epoch 200  --decay_epoch  150  --beta 1e+10   --lr 0.1 --eps 0.1 --model vgg_bn --optim exptamed_v0 --seed 111 &
srun --ntasks=1 --nodes=1 --exclusive python  main.py --dataset cifar10 --total_epoch 200  --decay_epoch  150  --beta 1e+10   --lr 0.1 --eps 0.1 --model vgg_bn --optim exptamed_v0 --seed 222 &
srun --ntasks=1 --nodes=1 --exclusive python  main.py --dataset cifar10 --total_epoch 200  --decay_epoch  150  --beta 1e+10   --lr 0.1 --eps 0.1 --model vgg_bn --optim exptamed_v0 --seed 333 &

wait
