import torch

# local
from utils.l2l import *
from utils.gan_loss import *

def eval_joint_adversarial(model, dataloader, opt):
    # loss objectives
    criterionGAN = GANLoss('vanilla').to(opt.device)
    criterionL1 = torch.nn.L1Loss()
    # unpack model
    generator, discriminator = model
    total_loss_D, total_loss_G, total_loss_G_L1 = 0, 0, 0
    for eval_batch in dataloader:
        # flatten first two axis - don't care about per event classification of different frames
        src_img = eval_batch['A'].view(-1, *(eval_batch['A'].size()[2:])).float()
        tgt_img = eval_batch['B'].view(-1, *(eval_batch['B'].size()[2:])).float()
        src_img, tgt_img = src_img.to(opt.device), tgt_img.to(opt.device)
        prediction = generator(src_img)
        resized_src_img = nn.Upsample(scale_factor=2, mode='bilinear') (src_img)
        # -------------------------------------------------------------------------------- discriminator loss
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
        # -------------------------------------------------------------------------------- keep track of loss over whole dataset
        total_loss_D += loss_D.item()
        total_loss_G += loss_G.item()
        total_loss_G_L1 += loss_G_L1.item()
    return total_loss_D / len(dataloader), total_loss_G / len(dataloader), total_loss_G_L1 / len(dataloader)

def eval_maml_adversarial(model, dataloader, opt):
    generator, discriminator = model
    loss_D, loss_G, loss_G_L1 = 0, 0, 0
    for eval_batch in dataloader:
        for task_idx in range(eval_batch['A'].size()[0]): #for each task
            task_data = {
                    'A': eval_batch['A'][task_idx, :, :, :, :],
                    'B': eval_batch['B'][task_idx, :, :, :, :],
                }
            task_loss_D, task_loss_G, task_loss_G_L1, _, _ = adapt_adversarial(model, opt, task_data)
            loss_D += task_loss_D.item() / opt.batch_size
            loss_G += task_loss_G.item() / opt.batch_size
            loss_G_L1 += task_loss_G_L1.item() / opt.batch_size
    return loss_D / len(dataloader), loss_G / len(dataloader), loss_G_L1 / len(dataloader)

def eval_joint_reconstruction(model, dataloader, opt):
    total_loss = 0
    loss_fn = torch.nn.L1Loss()
    for eval_batch in dataloader:
        # flatten first two axis - don't care about per event classification of different frames
        src_img = eval_batch['A'].view(-1, *(eval_batch['A'].size()[2:])).float()
        tgt_img = eval_batch['B'].view(-1, *(eval_batch['B'].size()[2:])).float()
        src_img, tgt_img = src_img.to(opt.device), tgt_img.to(opt.device)
        # compute loss
        prediction = model(src_img)
        loss = loss_fn(tgt_img, prediction)
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_maml_reconstruction(model, dataloader, opt):
    total_loss = 0
    for eval_batch in dataloader:
        # data is meta-batch of eval tasks - adapt and track loss on each
        for task_idx in range(eval_batch['A'].size()[0]): #for each task
            task_data = {
                    'A': eval_batch['A'][task_idx, :, :, :, :],
                    'B': eval_batch['B'][task_idx, :, :, :, :],
                    }
            task_loss = adapt_reconstruction(model, task_data, opt)
            total_loss += task_loss.item() / opt.batch_size
    return total_loss / len(dataloader)
