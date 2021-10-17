import yaml
import argparse

if __name__ == "__main__":
    # get experiment spec from yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    with open('experiments/test.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # string snippets to create filenames
    loss_fn = data['loss_function']
    opt = data['optimization']
    bsz = "bsz={}".format(data['batch_size'])
    sn = "use-sn={}".format(data['spectral_norm_discriminator'])
    pretrained = "ckpt=" + (data['encoder_checkpoint'].split("/")[-1].split(".")[0] if data['pretrained_encoder'] else "none")
    fn_bookkeeping = "{}_{}_{}_{}_{}".format(loss_fn, opt, bsz, sn, pretrained)

    # ------------------------------------------------------------------------------------------------ train script
    # script preamble
    preamble = "".join([
        "#!/bin/sh\n",
        "#SBATCH --cpus-per-task=40\n",
        "#SBATCH --gres=gpu:volta:2\n",
        "#SBATCH -o logs/train_{}.out\n".format(fn_bookkeeping),
        "#SBATCH --job-name=sevir-train-job\n",
        ])
    # train command
    command = "".join([
        "python -u train_unet.py \\\n",
        "       --multi_gpu \\\n",
        "       --batch_size={} \\\n".format(data['batch_size']),
        "       --n_support=10 \\\n",
        "       --n_query=10 \\\n",
        "       --loss_function={} \\\n".format(loss_fn),
        "       --optimization={} \\\n".format(opt),
        "       --resize_target \\\n",
        "       --target_size=384 \\\n",
        "       --device=cuda \\\n",
        "       --fraction_dataset=1 \\\n",
        "       --checkpoint={}/{}/ ".format(data['checkpoint_folder'], fn_bookkeeping)
    ])
    if data['spectral_norm_discriminator']:
        command += "\\\n       --spectral_norm_discriminator "
    if data['pretrained_encoder']:
        command += "\\\n       --pretrained_encoder \\\n       --encoder_checkpoint={}".format(data['encoder_checkpoint'])
    # put them together
    script = "\n".join([preamble, command])
    # save train script to file
    f = open("train-scripts/" + fn_bookkeeping + ".sh", "w")
    f.write(script)
    f.close()
    # ------------------------------------------------------------------------------------------------ eval script
    preamble = "".join([
        "#!/bin/sh\n",
        "#SBATCH --cpus-per-task=40\n",
        "#SBATCH --gres=gpu:volta:2\n",
        "#SBATCH -o logs/eval_{}.out\n".format(fn_bookkeeping),
        "#SBATCH --job-name=sevir-eval-job\n",
        ])
    command = "".join([
        "python -u eval_unet.py \\\n",
        "       --unet_checkpoint={}/{}/generator_last.pt \\\n".format(data['checkpoint_folder'], fn_bookkeeping),
        "       --path_to_opt={}/{}/opt.txt \\\n".format(data['checkpoint_folder'], fn_bookkeeping),
        "       --phase=valid"
        ])
    script = "\n".join([preamble, command])
    f = open("eval-scripts/" + fn_bookkeeping + ".sh", "w")
    f.write(script)
    f.close()
