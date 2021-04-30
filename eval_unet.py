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
from misc.scrape_logs import *

if __name__ == "__main__":
    # process single cli arg and load opt from train
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--log', type=str, required=True)
    opt = parser.parse_args()
    log_path = opt.log

    with open(os.path.join(opt.checkpoint, "opt.txt"), 'r') as f:
        opt.__dict__ = json.load(f)

    # find checkpoint to load
    steps, losses, lines = parse(log_path)
    best_step = np.argmin(losses)
    line = lines[best_step]
    print ("loading checkpoint: ", line)
    epoch = int(line[3][0:-1])
    step = int(line[5])

    # model
    model = Unet()
    model.load_state_dict(torch.load(os.path.join(opt.checkpoint, "checkpoint_epoch_{}_step_{}.pt".format(epoch, step))))
    model = model.to(opt.device)
    model.eval()

    # data
    opt.phase = "test"
    test_dataset = SevirDataset(opt)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size)

    print ("using {} test tasks".format(len(test_dataset)))

    # get metric functions and names
    metric_functions = get_metric_functions()
    metric_names =  get_metric_functions().keys()

    # setup results dict
    results = dict()
    for key in metric_names:
        results[key] = 0

    # compute metrics
    for count, test_batch in enumerate(test_dataloader):
        # flatten first two axis - don't care about per event classification of different frames
        src_img = test_batch['A'].view(-1, *(test_batch['A'].size()[2:])).float()
        tgt_img = test_batch['B'].view(-1, *(test_batch['B'].size()[2:])).float()
        # fwd pass
        src_img, tgt_img = src_img.to(opt.device), tgt_img.to(opt.device)
        prediction = model(src_img)
        # undo norm to use thresholds
        vil_mu, vil_sigma = test_dataset.znorm['vil']
        tgt_img = vil_mu + tgt_img * vil_sigma
        prediction = vil_mu + prediction * vil_sigma
        # compute metrics
        metrics = compute_metrics(tgt_img, prediction, metric_functions)
        for key in results:
            results[key] += metrics[key]

        if (count + 1) % 10 == 0:
            # norm by number of batches
            for key in results:
                print ("for key {} estimate after {} test batches is {}".format(key, count, results[key] / count))

    # norm by number of batches
    for key in results:
        results[key] /= len(test_dataloader)
    print (results)
    # save samples to disk
    np.save(os.path.join(opt.checkpoint, 'prediction'), prediction.detach().cpu().numpy())
    np.save(os.path.join(opt.checkpoint, 'target'), tgt_img.detach().cpu().numpy())



