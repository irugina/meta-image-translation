import time
import os

import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
import torch.nn as nn

# local imports
from data.sevir_dataset import SevirDataset
from models.sevir_generator import Unet
from models.sevir_discriminator import NLayerDiscriminator
from utils.train import *
from utils.eval import *
from utils.parse_args import *

if __name__ == "__main__":
    # parse cli args
    opt = parse_train_args()

    # U-Net generator
    generator = Unet()
    if opt.pretrained_encoder:
        # load pretained encoder
        save_dict = torch.load(opt.encoder_checkpoint, map_location='cpu')
        # filter for keys we have in generator
        saved_dict = {k[9:]: v for k, v in save_dict['state_dict'].items() if
                (k.startswith('backbone.') and ('inc' in k or 'down' in k))}
        # update state dixt
        generator_state = generator.state_dict()
        generator_state.update(saved_dict)
        generator.load_state_dict(generator_state)
    generator = generator.to(opt.device)
    if opt.multi_gpu: # on supercloud this will always be 2 gpus
        generator = nn.DataParallel(generator, [0,1])

    # discriminator if training adversarially
    if opt.loss_function == 'adversarial':
        # conditional GAN - discriminator takes in both src and tgt views
        discriminator = NLayerDiscriminator(opt.input_nc + opt.output_nc)
        discriminator = discriminator.to(opt.device)
        if opt.multi_gpu: # on supercloud this will always be 2 gpus
            discriminator = nn.DataParallel(discriminator, [0,1])
        model = (generator, discriminator)
    else:
        model = generator

    # data
    train_dataset = SevirDataset(opt)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=32)

    # train
    train_fn = eval("train_{}_{}".format(opt.optimization, opt.loss_function))
    for epoch in range(opt.n_epochs):
        t1 = time.time()
        train_fn(model, train_dataloader, opt, epoch)
        t2 = time.time()
        print ("one epoch took {} seconds".format(t2-t1))
    torch.save(generator.state_dict(), os.path.join(opt.checkpoint, "generator_last.pt"))
