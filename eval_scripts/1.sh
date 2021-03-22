#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/1_eval.out
#SBATCH --job-name=1_eval

python -u eval_unet.py \
	--checkpoint=reconstruction_joint/
