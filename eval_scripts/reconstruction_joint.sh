#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/second_run_reconstruction_joint_eval.out
#SBATCH --job-name=reconstruction_joint_eval

python -u eval_unet.py \
	--checkpoint=checkpoints/reconstruction_joint/ \
        --log=logs/reconstruction_joint_train.out
