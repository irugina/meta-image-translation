import time
import os

import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam

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

    # model
    model = Unet()
    if opt.pretrained_encoder:
        # load pretained encoder
        save_dict = torch.load(opt.encoder_checkpoint, map_location='cpu')
        # filter for keys we have in model
        saved_dict = {k[9:]: v for k, v in save_dict['state_dict'].items() if
                (k.startswith('backbone.') and ('inc' in k or 'down' in k))}
        # update model
        model_state = model.state_dict()
        model_state.update(saved_dict)
        model.load_state_dict(model_state)
    model = model.to(opt.device)

    # discriminator if training adversarially
    if opt.loss_function == 'adversarial':
        # conditional GAN - discriminator takes in both src and tgt views
        discriminator = NLayerDiscriminator(opt.input_nc + opt.output_nc)
        print (discriminator)
    else:
        discriminator = None

    # data
    train_dataset = SevirDataset(opt)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size)

    # hack to create eval dataloader in this script
    opt.phase = "valid"
    eval_dataset = SevirDataset(opt)
    eval_dataloader = DataLoader(eval_dataset, batch_size=opt.batch_size)
    opt.phase = "train"

    print ("{} train tasks".format(len(train_dataloader)))
    print ("{} eval tasks".format(len(eval_dataloader)))

    # train
    train_fn = eval("train_{}_{}".format(opt.optimization, opt.loss_function))
    for epoch in range(opt.n_epochs):
        t1 = time.time()
        train_fn(model, train_dataloader, eval_dataloader, opt, epoch)
        t2 = time.time()
        print ("one epoch took {} seconds".format(t2-t1))
    torch.save(model.state_dict(), os.path.join(opt.checkpoint, "checkpoint_last.pt"))
