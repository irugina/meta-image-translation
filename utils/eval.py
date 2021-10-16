import torch
# local
from utils.l2l import *
from utils.gan_loss import *
from utils.common_helpers import *

def eval_joint_adversarial(model, dataloader, opt):
    # loss objectives
    criterionGAN = GANLoss('vanilla').to(opt.device)
    criterionL1 = torch.nn.L1Loss()
    # unpack model
    generator, discriminator = model
    total_loss_D_fake, total_loss_D_real, total_loss_G_GAN, total_loss_G_L1 = 0, 0, 0, 0
    for eval_batch in dataloader:
        src_img, tgt_img = flatten_for_joint (eval_batch, opt.device)
        pred = generator(src_img)
        resized_src_img = nn.Upsample(scale_factor=2, mode='bilinear') (src_img)
        # compute losses
        loss_D_fake, loss_D_real, loss_G_GAN, loss_G_L1 = compute_GAN_loss(resized_src_img, tgt_img, pred, discriminator, criterionGAN, criterionL1, opt.lambda_L1)
        total_loss_D_fake += loss_D_fake.item(); total_loss_D_real += loss_D_real.item()
        total_loss_G_GAN += loss_G_GAN.item(); total_loss_G_L1 += loss_G_L1.item()
    result = dict()
    result['loss_D_fake'] = total_loss_D_fake; result['loss_D_real'] = total_loss_D_real
    result['loss_G_GAN'] = total_loss_G_GAN; result['loss_G_L1'] = total_loss_G_L1
    for key, value in result.items():
            value /= len(dataloader)
    return result

def eval_maml_adversarial(model, dataloader, opt):
    generator, discriminator = model
    loss_D, loss_G, loss_G_L1 = 0, 0, 0
    for eval_batch in dataloader:
        for task_idx in range(eval_batch['A'].size()[0]): #for each task
            # adapt to task using support set
            task_data = pack_for_maml(eval_batch, task_idx)
            _, _, task_discriminator, task_generator = adapt_adversarial(model, opt, task_data)
            # fwd pass on query set
            _, _, query_real_A , query_real_B = unpack_data(data, opt)
            pred = task_generator(query_real_A)
            resized_src_img = nn.Upsample(scale_factor=2, mode='bilinear') (query_real_A)
            # compute losses on query 
            loss_D_fake, loss_D_real, loss_G_GAN, loss_G_L1 = compute_GAN_loss(
                    resized_src_img, 
                    query_real_B, 
                    pred, 
                    task_discriminator, 
                    criterionGAN, 
                    criterionL1, 
                    opt.lambda_L1)
            total_loss_D_fake += loss_D_fake.item(); total_loss_D_real += loss_D_real.item()
            total_loss_G_GAN += loss_G_GAN.item(); total_loss_G_L1 += loss_G_L1.item()
    result = dict()
    result['loss_D_fake'] = total_loss_D_fake; result['loss_D_real'] = total_loss_D_real
    result['loss_G_GAN'] = total_loss_G_GAN; result['loss_G_L1'] = total_loss_G_L1
    for key, value in result.items():
            value /= len(dataloader) 
            value /= opt.batch_size
    return result

def eval_joint_reconstruction(model, dataloader, opt):
    total_loss = 0
    loss_fn = torch.nn.L1Loss()
    for eval_batch in dataloader:
        src_img, tgt_img = flatten_for_joint (eval_batch, opt.device)
        prediction = model(src_img)
        # compute loss
        loss = loss_fn(tgt_img, prediction)
        total_loss += loss_fn(tgt_img, prediction).item()
    result = dict()
    result['MAE'] = total_loss / len(dataloader)
    return result 

def eval_maml_reconstruction(model, dataloader, opt):
    total_loss = 0
    for eval_batch in dataloader:
        # data is meta-batch of eval tasks - adapt and track loss on each
        for task_idx in range(eval_batch['A'].size()[0]): #for each task
            task_data = pack_for_maml(eval_batch, task_idx)
            task_loss, task_model= adapt_reconstruction(model, task_data, opt)
            total_loss += task_loss.item() / opt.batch_size
    result = dict()
    result['MAE'] = total_loss / len(dataloader)
    return result 
