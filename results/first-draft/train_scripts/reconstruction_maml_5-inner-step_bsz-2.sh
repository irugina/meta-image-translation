#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/reconstruction_maml_5-inner-step_2-bsz_train.out
#SBATCH --job-name=reconstruction_maml_5-inner-step_2-bsz_train

python -u train_unet.py \
	--batch_size=2 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=reconstruction \
	--optimization=maml \
	--inner_steps=5 \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=250 \
	--fraction_dataset=1 \
	--checkpoint=reconstruction_maml_5-inner-step_2-bsz_train/
