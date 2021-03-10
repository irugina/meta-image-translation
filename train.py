import time
import os
import argparse

import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam

# local imports
from data.sevir_dataset import SevirDataset
from models.sevir import Unet

def train_joint_adversarial():
    pass

def train_maml_adversarial():
    pass

def train_joint_reconstruction(model, dataloader):
    loss_fn = torch.nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    for train_batch in dataloader:
        # flatten first two axis - don't care about per event classification of different frames
        src_img = train_batch['A'].view(-1, *(train_batch['A'].size()[2:])).float()
        tgt_img = train_batch['B'].view(-1, *(train_batch['B'].size()[2:])).float()
        # optimizer step
        prediction = model(src_img)
        optimizer.zero_grad()
        loss = loss_fn(tgt_img, prediction)
        loss.backward()
        optimizer.step()

def train_maml_reconstruction():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Train Arguments')

    # ------------------------------------------------------------------------------------------------training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--optimization', type=str, required=True)
    parser.add_argument('--loss_function', type=str, required=True)

    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=10, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

    # ------------------------------------------------------------------------------------------------data
    # filepaths
    parser.add_argument('--dataroot', type=str,
            default='/home/gridsan/groups/EarthIntelligence/datasets/SEVIR/image_translation/')
    parser.add_argument('--phase', type=str, default='train')
    # use these for fast prototyping
    parser.add_argument('--load_size', type=int, default=192)
    parser.add_argument('--crop_size', type=int, default=192)
    parser.add_argument('--fraction_dataset', type=int, default=1)
    parser.add_argument('--frames_per_event', type=int, default=49)
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

    # sanity checks for scripts
    assert opt.optimization in {'joint', 'maml'}
    assert opt.loss_function in {'reconstruction', 'adversarial'}
    if opt.resize_target:
        assert opt.target_size != -1

    # model
    model = Unet()

    # data
    dataset = SevirDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size)

    # train
    train = eval("train_{}_{}".format(opt.optimization, opt.loss_function))
    train(model, dataloader)
