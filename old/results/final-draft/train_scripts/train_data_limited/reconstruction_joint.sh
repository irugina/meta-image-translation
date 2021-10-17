#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/reconstruction_joint_train_10.out
#SBATCH --job-name=reconstruction_joint_train_10

python -u train_unet.py \
	--batch_size=4 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=reconstruction \
	--optimization=joint \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=25 \
	--fraction_dataset=10 \
	--checkpoint=checkpoint-low-data/reconstruction_joint_10/
