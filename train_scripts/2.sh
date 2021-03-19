#!/bin/sh
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:volta:2
#SBATCH -o logs/2.out
#SBATCH --job-name=2

python -u main.py \
	--batch_size=4 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=reconstruction \
	--optimization=maml \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=1000 \
	--fraction_dataset=1
