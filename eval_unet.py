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
from models.sevir import Unet
from utils.train import *
from utils.eval import *
from metrics.metric import *

if __name__ == "__main__":
    # process single cli arg and load opt from train
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    opt = parser.parse_args()
    with open(opt.checkpoint + "opt.txt", 'r') as f:
        opt.__dict__ = json.load(f)
    print (opt)

    # model
    model = Unet()
    model.load_state_dict(torch.load(opt.checkpoint + "checkpoint_500.pt")) #todo
    model = model.to(opt.device)
    model.eval()

    # data
    eval_dataset = SevirDataset(opt)
    eval_dataloader = DataLoader(eval_dataset, batch_size=opt.batch_size)

    # get metric functions and names
    metric_functions = get_metric_functions()
    metric_names =  get_metric_functions().keys()

    # setup results dict
    results = dict()
    for key in metric_names:
        results[key] = 0

    # compute metrics
    for eval_batch in eval_dataloader:
        # flatten first two axis - don't care about per event classification of different frames
        src_img = eval_batch['A'].view(-1, *(eval_batch['A'].size()[2:])).float()
        tgt_img = eval_batch['B'].view(-1, *(eval_batch['B'].size()[2:])).float()
        # fwd pass
        src_img, tgt_img = src_img.to(opt.device), tgt_img.to(opt.device)
        prediction = model(src_img)
        # undo norm to use thresholds
        vil_mu, vil_sigma = eval_dataset.znorm['vil']
        src_img = vil_mu + src_img * vil_sigma
        prediction = vil_mu + prediction * vil_sigma
        # compute metrics
        metrics = compute_metrics(tgt_img, prediction, metric_functions)
        for key in results:
            results[key] += metrics[key]
    # norm by number of batches
    for key in results:
        results[key] /= len(eval_dataloader)
    print (results)
    # save samples to disk
    np.save('prediction', prediction.detach().cpu().numpy())
    np.save('target', tgt_img.detach().cpu().numpy())



