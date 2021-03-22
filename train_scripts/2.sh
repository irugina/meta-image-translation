#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/2_train.out
#SBATCH --job-name=2_train

python -u train_unet.py \
	--batch_size=4 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=reconstruction \
	--optimization=maml \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=100 \
	--fraction_dataset=1 \
	--checkpoint=reconstruction_maml/
