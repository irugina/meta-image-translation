#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/reconstruction_maml_3-inner-step_train.out
#SBATCH --job-name=reconstruction_maml_3-inner-step_train

python -u train_unet.py \
	--batch_size=4 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=reconstruction \
	--optimization=maml \
	--inner_steps=3 \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=250 \
	--fraction_dataset=1 \
	--checkpoint=reconstruction_maml_3-inner-step_train/
