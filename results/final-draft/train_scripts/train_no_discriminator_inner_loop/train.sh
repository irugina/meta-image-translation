#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/adversarial_maml_freeze_D.out
#SBATCH --job-name=adversarial_maml_freeze_D

python -u train_unet.py \
	--fix_inner_loop_discriminator \
	--batch_size=2 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=adversarial \
	--optimization=maml \
	--inner_steps=1 \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=500 \
	--fraction_dataset=1 \
	--checkpoint=checkpoints/adversarial_maml_freeze_D
