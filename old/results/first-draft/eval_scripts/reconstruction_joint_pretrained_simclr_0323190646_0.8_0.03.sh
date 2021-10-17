#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/pretrained_reconstruction_joint_simclr_0323190646_0.8_0.03_eval.out
#SBATCH --job-name=pretrained_reconstruction_joint_simclr_0323190646_0.8_0.03_eval

python -u eval_unet.py \
	--checkpoint=pretained_reconstruction_joint_simclr_0323190646_0.8_0.03/ \
        --log=logs/pretrained_reconstruction_joint_simclr_0323190646_0.8_0.03_train.out
