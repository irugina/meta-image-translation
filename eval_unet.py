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

def intermediate_output(results, count, src_img, prediction, tgt_img):
    # evaluate weather metrics
    for key in results:
        print ("for key {} estimate after {} test batches is {}".format(key, count, results[key] / count))
    # save samples to disk
    np.save(os.path.join(opt.checkpoint, 'source_{}'.format(count)), src_img.detach().cpu().numpy())
    np.save(os.path.join(opt.checkpoint, 'prediction_{}'.format(count)), prediction.detach().cpu().numpy())
    np.save(os.path.join(opt.checkpoint, 'target_{}'.format(count)), tgt_img.detach().cpu().numpy())

def find_best_checkpoint(log_path):
    steps, losses, lines = parse(log_path)
    best_step = np.argmin(losses)
    line = lines[best_step]
    print ("loading checkpoint: ", line)
    epoch = int(line[3][0:-1])
    step = int(line[5])
    return epoch, step

def load_reconstruction_generator(opt, epoch, step):
    model = Unet()
    model.load_state_dict(torch.load(os.path.join(opt.checkpoint, "checkpoint_epoch_{}_step_{}.pt".format(epoch, step))))
    model.eval()
    return model

def load_adversarial_model(opt, epoch, step):
    # generator
    generator = Unet()
    generator.load_state_dict(torch.load(os.path.join(opt.checkpoint, "generator_epoch_{}_step_{}.pt".format(epoch, step))))
    generator.eval()
    # discriminator
    discriminator = NLayerDiscriminator(opt.input_nc + opt.output_nc)
    discriminator.load_state_dict(torch.load(os.path.join(opt.checkpoint, "discriminator_epoch_{}_step_{}.pt".format(epoch, step))))
    discriminator.eval()
    # return
    return generator, discriminator

def evaluate_prediction(vil_znorm, src_img, tgt_img, prediction, results, count):
    # undo norm to use thresholds
    vil_mu, vil_sigma = vil_znorm
    tgt_img = vil_mu + tgt_img * vil_sigma
    prediction = vil_mu + prediction * vil_sigma
    # compute metrics
    metrics = compute_metrics(tgt_img, prediction, metric_functions)
    for key in results:
        results[key] += metrics[key]
    if (count + 1) % 20 == 0:
        intermediate_output(results, count, src_img, prediction, tgt_img)

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
    epoch, step = find_best_checkpoint(log_path)

    if "checkpoint" not in opt.checkpoint:
        opt.checkpoint = os.path.join("checkpoints", opt.checkpoint)

    # model
    if opt.loss_function == "reconstruction":
        model = load_reconstruction_generator(opt, epoch, step)
    if opt.loss_function == "adversarial":
        generator, discriminator = load_adversarial_model(opt, epoch, step)

    # data
    opt.phase = "test"
    test_dataset = SevirDataset(opt)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # get metric functions and names
    metric_functions = get_metric_functions()
    metric_names =  get_metric_functions().keys()

    # setup results dict
    results = dict()
    for key in metric_names:
        results[key] = 0

    # zscore normalization
    vil_znorm = test_dataset.znorm['vil']

    # compute metrics
    for count, test_batch in enumerate(test_dataloader):
        if opt.optimization == "joint": #this is easy: no adaptatation
            # put unet on GPU - regardless of how it was trained
            if opt.loss_function == "reconstruction":
                model = model.to(opt.device)
            if opt.loss_function == "adversarial":
                model = generator.to(opt.device)
            # flatten first two axis - don't care about per event classification of different frames
            src_img = test_batch['A'].view(-1, *(test_batch['A'].size()[2:])).float()
            tgt_img = test_batch['B'].view(-1, *(test_batch['B'].size()[2:])).float()
            # fwd pass
            src_img, tgt_img = src_img.to(opt.device), tgt_img.to(opt.device)
            prediction = model(src_img)

        # NOTE code below makes sense iff test dataloader has batch_size == 1
        if opt.optimization == "maml" and opt.loss_function == "adversarial":
            generator, discriminator = generator.to(opt.device), discriminator.to(opt.device)
            model = (generator, discriminator)
            for task_idx in range(test_batch['A'].size()[0]): #for each test task
                task_data = {
                        'A': test_batch['A'][task_idx, :, :, :, :],
                        'B': test_batch['B'][task_idx, :, :, :, :],
                    }
                src_img, prediction, tgt_img = adapt_adversarial(model, opt, task_data, True)

        if opt.optimization == "maml" and opt.loss_function == "reconstruction":
            model = model.to(opt.device)
            for task_idx in range(test_batch['A'].size()[0]): #for each test task
                task_data = {
                        'A': test_batch['A'][task_idx, :, :, :, :].float(),
                        'B': test_batch['B'][task_idx, :, :, :, :].float(),
                    }
                src_img, prediction, tgt_img = adapt_reconstruction(model, task_data, opt, True)

        # eval weather metrics and save some samples to disk
        evaluate_prediction(vil_znorm, src_img, tgt_img, prediction, results, count)

    # norm by number of batches
    for key in results:
        results[key] /= len(test_dataloader)
    print (results)
