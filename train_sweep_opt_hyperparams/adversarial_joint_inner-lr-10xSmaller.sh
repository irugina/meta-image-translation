#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/adversarial_joint_train_inner-lr-10xSmaller.out
#SBATCH --job-name=adversarial_joint__train_inner-lr-10xSmaller

python -u train_unet.py \
	--batch_size=4 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=adversarial \
	--optimization=joint \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=250 \
	--fraction_dataset=1 \
	--checkpoint=checkpoints/adversarial_joint_inner-lr-10xSmaller/ \
	--inner_lr=0.00001
