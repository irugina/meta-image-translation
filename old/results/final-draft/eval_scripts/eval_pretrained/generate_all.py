#name = "simclr_bl_0_no_0.5_sr_0_crop_0_e_4"
names = ["simclr_bl_0.2_no_0_sr_0_crop_0_e_16", "simclr_bl_0_no_0_sr_0_crop_1_e_4", "simclr_bl_0_no_0_sr_3_crop_0_e_4", "simclr_bl_0.2_no_0_sr_0_crop_1_e_4", "simclr_bl_0_no_0.5_sr_0_crop_1_e_12", "simclr_bl_0_no_0.5_sr_3_crop_0_e_4", "simclr_bl_0_no_0_sr_3_crop_1_e_16"]

for name in names:
    s1 = "#!/bin/sh\n"
    s2 = "#SBATCH --cpus-per-task=20\n"
    s3 = "#SBATCH --gres=gpu:volta:1\n"
    s4 = "#SBATCH -o logs/eval_{}\n".format(name)
    s5 = "#SBATCH --job-name=eval_{}\n\n".format(name)

    s6 = "python -u eval_unet.py \\\n"
    s7 = "        --log=../MAML-Soljacic/rumen/sevir/logs/{} \\\n".format(name)
    s8 = "        --checkpoint=../MAML-Soljacic/rumen/sevir/checkpoints/{}/ \n".format(name)

    f = open("{}.sh".format(name), "w")
    f.write(s1+s2+s3+s4+s5+s6+s7+s8)
    f.close()
