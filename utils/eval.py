from utils.l2l import *

def eval_joint_adversarial():
    pass

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
            loss_D += task_loss_D.item(); loss_G += task_loss_G.item(); loss_G_L1 += task_loss_G_L1.item()
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
