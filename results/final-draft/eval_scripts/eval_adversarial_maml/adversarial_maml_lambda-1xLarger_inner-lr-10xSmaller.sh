#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/adversarial_maml_eval_lambda-1xLarger_inner-lr-10xSmaller.out
#SBATCH --job-name=adversarial_maml_lambda-1xLarger_inner-lr-10xSmaller_eval

python -u eval_unet.py \
	--log=logs/adversarial_maml_train_lambda-1xLarger_inner-lr-10xSmaller.out \
	--checkpoint=checkpoints/adversarial_maml_lambda-1xLarger_inner-lr-10xSmaller 
