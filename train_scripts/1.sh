#!/bin/sh
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:volta:2
#SBATCH -o logs/1.out
#SBATCH --job-name=debug

python -u main.py \
	--batch_size=2
	--loss_function=reconstruction \
	--optimization=joint \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=1000 \
	--fraction_dataset=1
