#!/bin/sh
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:volta:2
#SBATCH -o logs/train_seed-44_reconstruction_joint_bsz-4_use-sn-False_ckpt-none.out
#SBATCH --job-name=sevir-train-job

python -u train_unet.py \
       --seed=44 \
       --multi_gpu \
       --batch_size=4 \
       --n_support=10 \
       --n_query=10 \
       --loss_function=reconstruction \
       --optimization=joint \
       --resize_target \
       --target_size=384 \
       --device=cuda \
       --fraction_dataset=1 \
       --checkpoint=checkpoints/seed-44_reconstruction_joint_bsz-4_use-sn-False_ckpt-none/ 