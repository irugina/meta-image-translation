#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/adversarial_joint_eval.out
#SBATCH --job-name=adversarial_joint_eval

python -u eval_unet.py \
	--checkpoint=checkpoints/adversarial_joint/ \
        --log=logs/adversarial_joint_train.out
