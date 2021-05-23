#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/adversarial_maml_eval.out
#SBATCH --job-name=adversarial_maml_eval

python -u eval_unet.py \
	--checkpoint=checkpoints/adversarial_maml/ \
        --log=logs/adversarial_maml_train.out
