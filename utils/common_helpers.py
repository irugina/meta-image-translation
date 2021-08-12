import torch


def compute_GAN_loss(src, tgt, pred, discriminator, criterionGAN, criterionL1, lambda_L1):
    '''
    compute all terms in generator and discriminator loss functions for GAN
    '''
    fake_AB = torch.cat((src, pred), 1)
    pred_fake = discriminator(fake_AB.detach())
    loss_D_fake = criterionGAN(pred_fake, False)
    # real
    real_AB = torch.cat((src, tgt), 1)
    pred_real = discriminator(real_AB)
    loss_D_real = criterionGAN(pred_real, True)
    # -------------------------------------------------------------------------------- generator loss
    # gan loss
    loss_G_GAN = criterionGAN(pred_fake, True)
    # reconstruction loss
    loss_G_L1 = criterionL1(pred, tgt) * lambda_L1
    return loss_D_fake, loss_D_real, loss_G_GAN, loss_G_L1

def flatten_for_joint(meta_batch, device):
    '''
    flatten all tasks within meta-batch for joint
    '''
    src_img = meta_batch['A'].view(-1, *(meta_batch['A'].size()[2:])).float()
    tgt_img = meta_batch['B'].view(-1, *(meta_batch['B'].size()[2:])).float()
    src_img, tgt_img = src_img.to(device), tgt_img.to(device)
    return src_img, tgt_img

def pack_for_maml(meta_batch, task_idx):
    '''
    select single task from meta_batch for maml
    '''
    task_data = {
            'A': meta_batch['A'][task_idx, :, :, :, :],
            'B': meta_batch['B'][task_idx, :, :, :, :],
        }
    return task_data

def unpack_data(data, opt):
    real_A = data['A'].to(opt.device)
    real_B = data['B'].to(opt.device)
    support_real_A, support_real_B = real_A[0:opt.n_support, :, :, :], real_B[0:opt.n_support, :, :, :]
    query_real_A , query_real_B  = real_A[opt.n_support:opt.n_support+opt.n_query, :, :, :], real_B[opt.n_support:opt.n_support+opt.n_query, :, :, :]
    return support_real_A, support_real_B, query_real_A , query_real_B
