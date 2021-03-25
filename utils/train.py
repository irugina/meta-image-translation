import torch
from torch.optim import Adam

# local
from utils.eval import *
from utils.l2l import *

def make_checkpoint(model, opt, eval_dataloader, total_steps, count, epoch):
    torch.save(model.state_dict(), opt.checkpoint + "checkpoint_epoch_{}_step_{}.pt".format(epoch, count))
    eval_fn = eval("eval_{}_{}".format(opt.optimization, opt.loss_function))
    print ("evaluating...")
    eval_loss = eval_fn(model, eval_dataloader, opt)
    print ("loss at epoch {}, step {} out of {}, was {}".format(epoch, count, total_steps, eval_loss))


def train_joint_adversarial():
    pass

def train_maml_adversarial():
    pass

def train_joint_reconstruction(model, train_dataloader, eval_dataloader, opt, epoch):
    loss_fn = torch.nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
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
        # eval
        if count % opt.eval_freq == 0:
            make_checkpoint(model, opt, eval_dataloader, len(train_dataloader), count, epoch)

    # also save end of epoch checkpoint
    torch.save(model.state_dict(), opt.checkpoint + "checkpoint_epoch_{}.pt".format(epoch))
    eval_fn = eval("eval_{}_{}".format(opt.optimization, opt.loss_function))
    print ("evaluating...")
    eval_loss = eval_fn(model, eval_dataloader, opt)
    print ("loss at end of epoch {} was {}".format(epoch, eval_loss))

def train_maml_reconstruction(model, train_dataloader, eval_dataloader, opt, epoch):
    loss_fn = torch.nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    for count, train_batch in enumerate(train_dataloader):
       # data is meta-batch of train tasks - adapt and track loss on each
       loss = 0
       for task_idx in range(train_batch['A'].size()[0]): #for each train task
           task_data = {
                   'A': train_batch['A'][task_idx, :, :, :, :],
                   'B': train_batch['B'][task_idx, :, :, :, :],
                   }
           task_loss = adapt(model, task_data, opt)
           loss += task_loss
       # perform outer loop optimization
       loss /= opt.batch_size
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       # eval
       if count % opt.eval_freq == 0:
            make_checkpoint(model, opt, eval_dataloader, len(train_dataloader), count, epoch)
