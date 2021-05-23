#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/eval_simclr_bl_0.2_no_0_sr_0_crop_0_e_16
#SBATCH --job-name=eval_simclr_bl_0.2_no_0_sr_0_crop_0_e_16

python -u eval_unet.py \
        --log=../MAML-Soljacic/rumen/sevir/logs/simclr_bl_0.2_no_0_sr_0_crop_0_e_16 \
        --checkpoint=../MAML-Soljacic/rumen/sevir/checkpoints/simclr_bl_0.2_no_0_sr_0_crop_0_e_16/ 
