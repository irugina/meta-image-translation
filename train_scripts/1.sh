#!/bin/sh
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:volta:2
#SBATCH -o logs/1.out
#SBATCH --job-name=1

python -u main.py \
	--batch_size=4 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=reconstruction \
	--optimization=joint \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=100 \
	--fraction_dataset=1 \
	--checkpoint=reconstruction_joint/
