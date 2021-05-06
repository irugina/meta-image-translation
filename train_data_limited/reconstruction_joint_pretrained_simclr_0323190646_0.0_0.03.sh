#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/pretrained_reconstruction_joint_simclr_0323190646_0.0_0.03_train_10.out
#SBATCH --job-name=pretrained_reconstruction_joint_simclr_0323190646_0.0_0.03_train_10

python -u train_unet.py \
	--batch_size=4 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=reconstruction \
	--optimization=joint \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=25 \
	--fraction_dataset=10 \
	--checkpoint=checkpoint-low-data/pretained_reconstruction_joint_simclr_0323190646_0.0_0.03_10/ \
	--pretrained_encoder \
	--encoder_checkpoint=/home/gridsan/groups/MAML-Soljacic/sevir_checkpoints/sevir_simclr_0323190646/0.0_0.03_epoch_700.pth
