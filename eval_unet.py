import time
import os
import argparse
import json

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
from metrics.metric import *
from misc.scrape_logs import *


def load_reconstruction_generator(opt, checkpoint):
    model = Unet()
    if opt.multi_gpu:
        model = model.to(opt.device)
        model = nn.DataParallel(model, [0,1])
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    return model

def load_adversarial_model(opt, unet_checkpoint, discriminator_checkpoint):
    # generator
    generator = Unet()
    if opt.multi_gpu:
        generator = generator.to(opt.device)
        generator = nn.DataParallel(generator, [0,1])
    generator.load_state_dict(torch.load(unet_checkpoint))
    generator.eval()
    # discriminator
    discriminator = NLayerDiscriminator(opt.input_nc + opt.output_nc)
    if opt.multi_gpu:
        discriminator = discriminator.to(opt.device)
        discriminator = nn.DataParallel(discriminator, [0,1])
    discriminator.load_state_dict(torch.load(discriminator_checkpoint))
    discriminator.eval()
    # return
    return generator, discriminator

def load_model(opt, unet_checkpoint, discriminator_checkpoint):
    if opt.loss_function == "adversarial":
        assert opt.discriminator_checkpoint is not None
        return load_adversarial_model(opt, unet_checkpoint, discriminator_checkpoint)
    return load_reconstruction_generator(opt, unet_checkpoint)


def evaluate_prediction(vil_znorm, tgt_img, prediction, results):
    # undo norm to use thresholds
    vil_mu, vil_sigma = vil_znorm
    tgt_img = vil_mu + tgt_img * vil_sigma
    prediction = vil_mu + prediction * vil_sigma
    # compute metrics
    metrics = compute_metrics(tgt_img, prediction, metric_functions)
    for key in results:
        results[key] += metrics[key]


if __name__ == "__main__":
    # process cli arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet_checkpoint', type=str, required=True)
    parser.add_argument('--discriminator_checkpoint', type=str, default=None)
    parser.add_argument('--path_to_opt', type=str, required=True)
    parser.add_argument('--phase', type=str, required=True)
    opt = parser.parse_args()
    # save cli args
    unet_checkpoint = opt.unet_checkpoint
    discriminator_checkpoint = opt.discriminator_checkpoint
    path_to_opt = opt.path_to_opt
    phase = opt.phase
    # load experiment opt from disk
    with open(path_to_opt,  'r') as f:
        opt.__dict__ = json.load(f)

    # dataset and loader
    opt.phase = phase
    dataset = SevirDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size)

    # model
    model = load_model(opt, unet_checkpoint, discriminator_checkpoint)

    # run eval
    eval_fn = eval("eval_{}_{}".format(opt.optimization, opt.loss_function))
    result = eval_fn (model, dataloader, opt)
    print (result)

    # get metric functions and names
    metric_functions = get_metric_functions()
    metric_names =  get_metric_functions().keys()

    # setup results dict
    results = dict()
    for key in metric_names:
        results[key] = 0

    # zscore normalization
    vil_znorm = dataset.znorm['vil']

    for count, batch in enumerate(dataloader):
        if opt.optimization == "joint":
            if opt.loss_function == "reconstruction":
                model = model.to(opt.device)
            if opt.loss_function == "adversarial":
                model = generator.to(opt.device)
            src_img, tgt_img = flatten_for_joint(batch, opt.device)
            prediction = model(src_img)
            evaluate_prediction(vil_znorm, tgt_img, prediction, results)
            print (result)
