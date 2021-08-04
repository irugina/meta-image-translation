import numpy as np
import torch
from torch.optim import Adam

# local
from utils.l2l import *
from utils.gan_loss import *

import os

def make_checkpoint_reconstruction(model, opt, count, epoch):
    if count  == -1:
        torch.save(model.state_dict(), os.path.join(opt.checkpoint, "checkpoint_epoch_{}.pt".format(epoch)))
    else:
        torch.save(model.state_dict(), os.path.join(opt.checkpoint, "checkpoint_epoch_{}_step_{}.pt".format(epoch, count)))


def make_checkpoint_adversarial(model, opt, count, epoch):
    generator, discriminator = model
    if count  == -1:
        torch.save(generator.state_dict(), os.path.join(opt.checkpoint, "generator_epoch_{}.pt".format(epoch)))
        torch.save(discriminator.state_dict(), os.path.join(opt.checkpoint, "discriminator_epoch_{}.pt".format(epoch)))
    else:
        torch.save(generator.state_dict(), os.path.join(opt.checkpoint, "generator_epoch_{}_step_{}.pt".format(epoch, count)))
        torch.save(discriminator.state_dict(), os.path.join(opt.checkpoint, "discriminator_epoch_{}_step_{}.pt".format(epoch, count)))



def train_joint_adversarial(model, train_dataloader,  opt, epoch):
    generator, discriminator = model
    # loss objectives
    criterionGAN = GANLoss('vanilla').to(opt.device)
    criterionL1 = torch.nn.L1Loss()
    # optimizers for generator and discriminator, respectively
    optimizer_G = Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    ckpts = np.linspace(0, len(train_dataloader) , num=10, endpoint=False, dtype=int)
    for count, train_batch in enumerate(train_dataloader):
        # flatten first two axis - don't care about per event classification of different frames
        src_img = train_batch['A'].view(-1, *(train_batch['A'].size()[2:])).float()
        tgt_img = train_batch['B'].view(-1, *(train_batch['B'].size()[2:])).float()
        src_img, tgt_img = src_img.to(opt.device), tgt_img.to(opt.device)
        prediction = generator(src_img)
        # -------------------------------------------------------------------------------- discriminator loss
        resized_src_img = nn.Upsample(scale_factor=2, mode='bilinear') (src_img)
        # fake
        fake_AB = torch.cat((resized_src_img, prediction), 1)
        pred_fake = discriminator(fake_AB.detach()) # stop backprop to the generator by detaching fake_B
        loss_D_fake = criterionGAN(pred_fake, False)
        # real
        real_AB = torch.cat((resized_src_img, tgt_img), 1)
        pred_real = discriminator(real_AB)
        loss_D_real = criterionGAN(pred_real, True)
        # combine loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        # -------------------------------------------------------------------------------- generator loss
        # gan loss
        loss_G_GAN = criterionGAN(pred_fake, True)
        # reconstruction loss
        loss_G_L1 = criterionL1(prediction, tgt_img) * opt.lambda_L1
        # combine loss
        loss_G = loss_G_GAN + loss_G_L1
        # -------------------------------------------------------------------------------- bwd pass
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        loss_D.backward(retain_graph=True)
        loss_G.backward()
        optimizer_D.step()
        optimizer_G.step()
        # ckpt
        if count in ckpts:
            print ('ckpt at step {} out of {} in epoch {}'.format(count, len(train_dataloader), epoch))
            make_checkpoint_adversarial(model, opt, count, epoch)

def train_maml_adversarial(model, train_dataloader, opt, epoch):
    generator, discriminator = model
    # optimizers for generator and discriminator, respectively
    optimizer_G = Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    ckpts = np.linspace(0, len(train_dataloader) , num=10, endpoint=False, dtype=int)
    for count, train_batch in enumerate(train_dataloader):
        loss_D, loss_G = 0,0
        for task_idx in range(train_batch['A'].size()[0]): #for each train task
            task_data = {
                    'A': train_batch['A'][task_idx, :, :, :, :],
                    'B': train_batch['B'][task_idx, :, :, :, :],
                }
            task_loss_D, task_loss_G, _, _, _ = adapt_adversarial(model, opt, task_data)
            loss_D += task_loss_D; loss_G += task_loss_G
        # perform outer loop
        loss_D /= opt.batch_size
        loss_G /= opt.batch_size
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        loss_D.backward(retain_graph=True)
        loss_G.backward()
        optimizer_D.step()
        optimizer_G.step()
        # ckpt
        if count in ckpts:
            print ('ckpt at step {} out of {} in epoch {}'.format(count, len(train_dataloader), epoch))
            make_checkpoint_adversarial(model, opt,  count, epoch)

def train_joint_reconstruction(model, train_dataloader,  opt, epoch):
    loss_fn = torch.nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    ckpts = np.linspace(0, len(train_dataloader) , num=10, endpoint=False, dtype=int)
    for count, train_batch in enumerate(train_dataloader):
        # flatten first two axis - don't care about per event classification of different frames
        src_img = train_batch['A'].view(-1, *(train_batch['A'].size()[2:])).float()
        tgt_img = train_batch['B'].view(-1, *(train_batch['B'].size()[2:])).float()
        src_img, tgt_img = src_img.to(opt.device), tgt_img.to(opt.device)
        # optimizer step
        prediction = model(src_img)
        optimizer.zero_grad()
        loss = loss_fn(tgt_img, prediction)
        loss.backward()
        optimizer.step()
        # ckpt
        if count in ckpts:
            print ('ckpt at step {} out of {} in epoch {}'.format(count, len(train_dataloader), epoch))
            make_checkpoint_reconstruction(model, opt, count, epoch)

def train_maml_reconstruction(model, train_dataloader, opt, epoch):
    loss_fn = torch.nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    ckpts = np.linspace(0, len(train_dataloader) , num=10, endpoint=False, dtype=int)
    print (ckpts)
    for count, train_batch in enumerate(train_dataloader):
        # data is meta-batch of train tasks - adapt and track loss on each
        loss = 0
        for task_idx in range(train_batch['A'].size()[0]): #for each train task
           task_data = {
                   'A': train_batch['A'][task_idx, :, :, :, :],
                   'B': train_batch['B'][task_idx, :, :, :, :],
                   }
           task_loss = adapt_reconstruction(model, task_data, opt)
           loss += task_loss
        # perform outer loop optimization
        loss /= opt.batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ckpt
        if count in ckpts:
            print ('ckpt at step {} out of {} in epoch {}'.format(count, len(train_dataloader), epoch))
            make_checkpoint_reconstruction(model, opt, count, epoch)
