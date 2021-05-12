#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/adversarial_maml_train_lambda-100xLarger_inner-lr-1xSmaller.out
#SBATCH --job-name=adversarial_maml_lambda-100xLarger_inner-lr-1xSmaller_train

python -u train_unet.py \
	--batch_size=4 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=adversarial \
	--optimization=maml \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=250 \
	--fraction_dataset=1 \
	--checkpoint=checkpoints/adversarial_maml_lambda-100xLarger_inner-lr-1xSmaller \
	--inner_lr=0.0001 \
	--lambda_L1=10000
