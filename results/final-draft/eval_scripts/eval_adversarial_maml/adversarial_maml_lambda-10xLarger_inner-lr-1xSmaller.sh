#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/adversarial_maml_eval_lambda-10xLarger_inner-lr-1xSmaller.out
#SBATCH --job-name=adversarial_maml_lambda-10xLarger_inner-lr-1xSmaller_eval

python -u eval_unet.py \
	--log=logs/adversarial_maml_train_lambda-10xLarger_inner-lr-1xSmaller.out \
	--checkpoint=checkpoints/adversarial_maml_lambda-10xLarger_inner-lr-1xSmaller \
