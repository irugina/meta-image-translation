#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/eval_simclr_bl_0_no_0.5_sr_0_crop_1_e_12
#SBATCH --job-name=eval_simclr_bl_0_no_0.5_sr_0_crop_1_e_12

python -u eval_unet.py \
        --log=../MAML-Soljacic/rumen/sevir/logs/simclr_bl_0_no_0.5_sr_0_crop_1_e_12 \
        --checkpoint=../MAML-Soljacic/rumen/sevir/checkpoints/simclr_bl_0_no_0.5_sr_0_crop_1_e_12/ 
