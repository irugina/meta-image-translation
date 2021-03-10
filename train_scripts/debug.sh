#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o debug.out
#SBATCH --job-name=debug

python train.py \
	--loss_function=reconstruction \
	--optimization=joint \
        --resize_target \
        --target_size=384 \
	--device=cuda
