#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/adversarial_maml_freeze_D_eval.out
#SBATCH --job-name=adversarial_maml_freeze_D

python -u eval_unet.py \
        --log=logs/adversarial_maml_freeze_D.out \
	--checkpoint=checkpoints/adversarial_maml_freeze_D
