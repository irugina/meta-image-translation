#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/reconstruction_maml_1-inner-step_2-bsz_eval.out
#SBATCH --job-name=reconstruction_maml_1-inner-step_2-bsz_eval

python -u eval_unet.py \
	--log=logs/reconstruction_maml_1-inner-step_2-bsz_train.out \
	--checkpoint=checkpoints/reconstruction_maml_1-inner-step_2-bsz_train/
