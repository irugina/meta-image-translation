#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/reconstruction_maml_train_inner-lr-10xSmaller.out
#SBATCH --job-name=reconstruction_maml_train_inner-lr-10xSmaller

python -u train_unet.py \
	--batch_size=2 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=reconstruction \
	--optimization=maml \
	--inner_steps=1 \
	--resize_target \
	--target_size=384 \
	--device=cuda \
	--eval_freq=500 \
	--fraction_dataset=1 \
	--checkpoint=checkpoints/reconstruction_maml_train_inner-lr-10xSmaller/ \
	--inner_lr=0.00001
