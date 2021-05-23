#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/adversarial_joint_eval_lambda-100xLarger.out
#SBATCH --job-name=adversarial_joint_eval_lambda-100xLarger

python -u eval_unet.py \
	--log=logs/adversarial_joint_train_lambda-100xLarger.out \
	--checkpoint=checkpoints/adversarial_joint_lambda-100xLarger
