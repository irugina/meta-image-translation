#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH -o logs/pretrained_reconstruction_joint_train_simsiam_0323191233_1.0_0.03.out
#SBATCH --job-name=pretrained_reconstruction_joint_train_simsiam_0323191233_1.0_0.03

python -u train_unet.py \
	--batch_size=4 \
	--n_support=10 \
	--n_query=10 \
	--loss_function=reconstruction \
	--optimization=joint \
        --resize_target \
        --target_size=384 \
	--device=cuda \
	--eval_freq=250 \
	--fraction_dataset=1 \
	--checkpoint=pretained_reconstruction_joint_simsiam_0323191233_1.0_0.03/ \
	--pretrained_encoder \
	--encoder_checkpoint=/home/gridsan/groups/MAML-Soljacic/sevir_checkpoints/sevir_simsiam_0323191233/1.0_0.03_epoch_700.pth
