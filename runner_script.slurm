#!/bin/bash

#SBATCH --account=uerdogan
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name=resnet18  # Job name



#SBATCH --partition=akya-cuda
#SBATCH -N 4                       # Request 4 nodes (1 node per optimizer)
#SBATCH --ntasks=4                 # 4 tasks total (1 per node)
#SBATCH --cpus-per-task=10 
#SBATCH --gres=gpu:1               # 1 GPU per node
#SBATCH --time=02:30:00             # Time limit per task
#SBATCH --output=optimizer_%j_%t.out  # Output file for each task

# Load necessary modules

module load apps/truba-ai/gpu-2024.0

conda activate gpu-2024.0

# Define optimizer tasks
#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.0001   --model efficient_b1 --optim adabelief --seed 333 --batchsize 128 &
#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.0001   --model efficient_b1 --optim adam --seed 333 --batchsize 128 &
#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.01 --eps 0.05 --beta 1e+10  --model efficient_b1 --optim theopoula --seed 111 --batchsize 128 &
#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.01 --eps 0.05 --beta 1e+10  --model efficient_b1 --optim exptamed_v0 --seed 111 --batchsize 128 &
#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.01 --eps 0.05 --beta 1e+10  --model efficient_b1 --optim theopoula --seed 222 --batchsize 128 &
#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.01 --eps 0.05 --beta 1e+10  --model efficient_b1 --optim exptamed_v0 --seed 222 --batchsize 128 &


#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.001   --model resnet18 --optim adabelief --seed 222 --batchsize 128 &
#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.001   --model resnet18 --optim adabelief --seed 333 --batchsize 128 &

srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.005 --eps 0.1 --beta 1e+10  --model resnet18 --optim exptamed_v0 --seed 111 --batchsize 128 &
srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.005 --eps 0.1 --beta 1e+10  --model resnet18 --optim theopoula --seed 111 --batchsize 128 &
#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.005 --eps 0.1 --beta 1e+11  --model resnet18 --optim exptamed_v0 --seed 222 --batchsize 128 &
#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.005 --eps 0.1 --beta 1e+11  --model resnet18 --optim theopoula --seed 222 --batchsize 128 &
srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.0001   --model resnet18 --optim adam --seed 111 --batchsize 128 &
srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.0001   --model resnet18 --optim adabelief --seed 111 --batchsize 128 &
#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.0001   --model resnet18 --optim adam --seed 333 --batchsize 128 &
#srun --ntasks=1 --nodes=1 --exclusive python  main_seed_loss.py   --total_epoch 20  --decay_epoch  150     --lr 0.0001   --model resnet18 --optim adabelief --seed 333 --batchsize 128 &

wait
