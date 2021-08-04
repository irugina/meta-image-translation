import os
import argparse
import json

def parse_train_args():
    parser = argparse.ArgumentParser(description='Process Train Arguments')

    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--multi_gpu', action='store_true')
    # ------------------------------------------------------------------------------------------------training
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--optimization', type=str, required=True)
    parser.add_argument('--loss_function', type=str, required=True)
    parser.add_argument('--n_epochs', type=int, default=2, help='number of epochs with the initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--eval_freq', type=int, required=True)
    parser.add_argument('--lambda_L1', type=float, default=100)
    # MAML
    parser.add_argument('--inner_steps', type=int, default='1')
    parser.add_argument('--inner_lr', type=float, default=0.0001)
    parser.add_argument('--first_order', action='store_true')
    parser.add_argument('--allow_unused', action='store_true')
    parser.add_argument('--allow_nograd', action='store_true')
    # save models and sample images to disk
    parser.add_argument('--checkpoint', type=str, required=True)
    # load pretrained encoder
    parser.add_argument('--pretrained_encoder', action='store_true')
    parser.add_argument('--encoder_checkpoint', type=str, default='')
    # freeze discriminator during MAML inner loop?
    parser.add_argument('--fix_inner_loop_discriminator', action='store_true')
    # ------------------------------------------------------------------------------------------------data
    # filepaths
    parser.add_argument('--dataroot', type=str,
            default='/home/gridsan/groups/EarthIntelligence/datasets/SEVIR/image_translation/')
    parser.add_argument('--phase', type=str, default='train')
    # use these for fast prototyping
    parser.add_argument('--load_size', type=int, default=192)
    parser.add_argument('--crop_size', type=int, default=192)
    parser.add_argument('--fraction_dataset', type=int, default=1)
    # translation setup details
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=1)
    # use these to support input and target of different size
    parser.add_argument('--resize_target', action="store_true")
    parser.add_argument('--target_size', type=int, default=-1)
    # use these for fast prototyping and/or experiment with the k in 'k-shot'
    parser.add_argument('--n_support', type=int, default=24)
    parser.add_argument('--n_query', type=int, default=25)

    opt = parser.parse_args()
    print (opt)
    with open(os.path.join(opt.checkpoint, "opt.txt"), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    # sanity checks for scripts
    assert opt.optimization in {'joint', 'maml'}
    assert opt.loss_function in {'reconstruction', 'adversarial'}
    if opt.resize_target:
        assert opt.target_size != -1
    if opt.pretrained_encoder:
        assert opt.encoder_checkpoint != ''

    return opt
