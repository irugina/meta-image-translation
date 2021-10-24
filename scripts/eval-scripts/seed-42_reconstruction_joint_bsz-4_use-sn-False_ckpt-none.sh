#!/bin/sh
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:volta:2
#SBATCH -o logs/eval_seed-42_reconstruction_joint_bsz-4_use-sn-False_ckpt-none.out
#SBATCH --job-name=sevir-eval-job

python -u eval_unet.py \
       --unet_checkpoint=checkpoints/seed-42_reconstruction_joint_bsz-4_use-sn-False_ckpt-none/generator_last.pt \
       --path_to_opt=checkpoints/seed-42_reconstruction_joint_bsz-4_use-sn-False_ckpt-none/opt.txt \
       --phase=valid