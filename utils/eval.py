import torch
from torch.optim import Adam

def eval_joint_adversarial():
    pass

def eval_maml_adversarial():
    pass

def eval_joint_reconstruction(model, dataloader, device):
    total_loss = 0
    loss_fn = torch.nn.L1Loss()
    for eval_batch in dataloader:
        # flatten first two axis - don't care about per event classification of different frames
        src_img = eval_batch['A'].view(-1, *(eval_batch['A'].size()[2:])).float()
        tgt_img = eval_batch['B'].view(-1, *(eval_batch['B'].size()[2:])).float()
        src_img, tgt_img = src_img.to(device), tgt_img.to(device)
        # compute loss
        prediction = model(src_img)
        loss = loss_fn(tgt_img, prediction)
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_maml_reconstruction():
    pass

