import time
import os
import argparse

import numpy as np
from torch.utils.data import DataLoader

# local imports
from data.sevir_dataset import SevirDataset
from models.sevir import Unet

def train_joint_adversarial():
    pass

def train_maml_adversarial():
    pass

def train_joint_reconstruction(model, dataloader):
    print (model)
    print (dataloader)

def train_maml_reconstruction():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Train Arguments')

    # optimization
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--optimization', type=str, required=True)
    parser.add_argument('--loss_function', type=str, required=True)
    # data
    parser.add_argument('--dataroot', type=str,
                        default='/home/gridsan/groups/EarthIntelligence/datasets/SEVIR/image_translation/')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--load_size', type=int, default=192)
    parser.add_argument('--crop_size', type=int, default=192)
    parser.add_argument('--fraction_dataset', type=int, default=1)
    parser.add_argument('--frames_per_event', type=int, default=49)
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=1)

    opt = parser.parse_args()

    # sanity checks for scripts
    assert opt.optimization in {'joint', 'maml'}
    assert opt.loss_function in {'reconstruction', 'adversarial'}

    # model
    model = Unet()

    # data
    dataset = SevirDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size)

    # train
    train = eval("train_{}_{}".format(opt.optimization, opt.loss_function))
    train(model, dataloader)
