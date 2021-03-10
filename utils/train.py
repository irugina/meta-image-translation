import torch
from torch.optim import Adam

# local
from utils.eval import *

def train_joint_adversarial():
    pass

def train_maml_adversarial():
    pass

def train_joint_reconstruction(model, train_dataloader, eval_dataloader, device, opt):
    loss_fn = torch.nn.L1Loss()
    eval_fn = eval("eval_{}_{}".format(opt.optimization, opt.loss_function))
    optimizer = Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    for count, train_batch in enumerate(train_dataloader):
        # flatten first two axis - don't care about per event classification of different frames
        src_img = train_batch['A'].view(-1, *(train_batch['A'].size()[2:])).float()
        tgt_img = train_batch['B'].view(-1, *(train_batch['B'].size()[2:])).float()
        src_img, tgt_img = src_img.to(device), tgt_img.to(device)
        # optimizer step
        prediction = model(src_img)
        optimizer.zero_grad()
        loss = loss_fn(tgt_img, prediction)
        loss.backward()
        optimizer.step()
        if count % opt.eval_freq == 0:
            print ("evaluating...")
            print ("loss = ", eval_fn(model, eval_dataloader, device))

def train_maml_reconstruction():
    pass

