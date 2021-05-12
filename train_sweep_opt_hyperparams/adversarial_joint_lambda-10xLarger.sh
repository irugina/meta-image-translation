#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/adversarial_joint_train_lambda-10xLarger.out
#SBATCH --job-name=adversarial_joint_train_lambda-10xLarger

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
	--checkpoint=checkpoints/adversarial_joint_lambda-10xLarger \
	--lambda_L1=1000
