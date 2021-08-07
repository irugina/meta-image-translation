#!/bin/sh
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:volta:2
#SBATCH -o logs/reconstruction_joint_train.out
#SBATCH --job-name=reconstruction_joint_train

python -u train_unet.py \
	--multi_gpu \
	--batch_size=8 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=reconstruction \
	--optimization=joint \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--fraction_dataset=1 \
	--checkpoint=checkpoints/reconstruction_joint/
