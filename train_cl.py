import os
import torch
import argparse
import time
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

from data.cl_sevir_dataset import SevirDataset
from models.sevir_generator import UnetEncoder


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class LRScheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate(
            (warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


def nt_xent_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape
    device = z1.device
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2 * N, dtype=torch.bool, device=device)
    diag[N:, :N] = diag[:N, N:] = diag[:N, :N]

    negatives = similarity_matrix[~diag].view(2 * N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2 * N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)


def negative_cosine_similarity_loss(p, z):
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


def info_nce_loss(nn, p, temperature=0.1):  # from NNCLR
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)

    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


class ContrastiveLearningTransform:
    def __init__(self, args):
        transforms = []
        if args.level == 0:
            transforms.append(T.RandomResizedCrop(size=192, scale=(1.0, 1.0)))
        else:
            transforms.append(T.RandomResizedCrop(size=192, scale=(0.8, 1.0)))
        if args.level > 1:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
        if args.level > 2:
            transforms.append(T.GaussianBlur(19))
        if args.level > 3:
            transforms.append(T.RandomApply(
                [T.Lambda(lambda x: x + 0.1 * torch.randn_like(x))], p=0.5))
        if args.level > 4:
            transforms.append(T.RandomVerticalFlip(p=0.2))
        if args.level > 5:
            transforms.append(T.RandomRotation(30))
        transforms.append(T.Normalize(
            (-3.64035706e+03, -1.43816570e+03, 3.09813984e-02),
            (1.17885038e+03, 2.59019927e+03, 5.13092746e-01)
        ))

        self.transform = T.Compose(transforms)

    def __call__(self, x):
        return self.transform(x)


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=False)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Branch(nn.Module):
    def __init__(self, proj_dim, proj_hidden, K=72 * 500, m=0.999):
        super().__init__()
        self.backbone = UnetEncoder()
        self.encoder_q = nn.Sequential(
            self.backbone,
            ProjectionMLP(256, proj_hidden, proj_dim)
        )
        self.encoder_k = nn.Sequential(
            UnetEncoder(),
            ProjectionMLP(256, proj_hidden, proj_dim)
        )
        self.register_buffer("queue", torch.randn(128, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.predictor = PredictionMLP(proj_dim, proj_hidden, proj_dim)
        self.K = K
        self.m = m

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x):
        return self.predictor(self.encoder_q(x)), self.encoder_k(x)


def ssl_loop(args):
    # logging
    os.makedirs(args.path_dir, exist_ok=True)
    file_to_update = open(os.path.join(args.path_dir, 'training.log'), 'w')

    # dataset
    dataloader_kwargs = dict(drop_last=True, pin_memory=True, num_workers=32)

    train_loader = torch.utils.data.DataLoader(
        dataset=SevirDataset(
            '/home/gridsan/groups/EarthIntelligence/datasets/SEVIR/image_translation/',
            'train',
            load_size=192,
            crop_size=192,
            frames_per_event=49,
            fraction_dataset=1,
            tgt_size=384,
            transform=ContrastiveLearningTransform(args),
        ),
        shuffle=True,
        batch_size=args.bsz,
        **dataloader_kwargs
    )

    # models
    dim_proj = [int(x) for x in args.dim_proj.split(',')]
    main_branch = Branch(dim_proj[1], dim_proj[0]).cuda()

    # optimization
    optimizer = torch.optim.SGD(
        main_branch.parameters(),
        momentum=0.9,
        lr=args.lr * args.bsz / 256,
        weight_decay=args.wd
    )
    lr_scheduler = LRScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=0,
        num_epochs=args.epochs,
        base_lr=args.lr * args.bsz / 256,
        final_lr=0,
        iter_per_epoch=len(train_loader)
    )

    # logging
    start = time.time()
    os.makedirs(args.path_dir, exist_ok=True)
    torch.save(dict(epoch=0, state_dict=main_branch.state_dict()),
               os.path.join(args.path_dir, '0.pth'))

    # training
    for e in range(1, args.epochs + 1):
        # declaring train
        main_branch.train()

        # epoch
        start = time.time()
        for it, inputs in enumerate(train_loader):
            # zero grad
            main_branch.zero_grad()
            x1 = inputs[0].view(-1, 3, 192, 192).cuda()
            x2 = inputs[1].view(-1, 3, 192, 192).cuda()

            main_branch.momentum_update_key_encoder()
            p1, k1 = main_branch(x1)
            p2, k2 = main_branch(x2)
            loss = info_nce_loss(k1.detach(), p2) / 2 + \
                info_nce_loss(k2.detach(), p1) / 2
            main_branch.dequeue_and_enqueue(k1)
            main_branch.dequeue_and_enqueue(k2)

            # # optimization step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            line_to_print = (
                f'epoch: {(e - 1 + it / len(train_loader)):.3f}/{e}  | loss: {loss.item():.3f} | time_elapsed: {time.time() - start:.3f}'
            )
            if file_to_update:
                file_to_update.write(line_to_print + '\n')
                file_to_update.flush()
            print(line_to_print)

        if e % args.save_every == 0:
            torch.save(dict(epoch=0, state_dict=main_branch.state_dict()),
                       os.path.join(args.path_dir, f'{e}.pth'))

    return main_branch.encoder


def main(args):
    fix_seed(42)
    ssl_loop(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_proj', default='2048,128', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.015, type=float)
    parser.add_argument('--bsz', default=3, type=int)
    parser.add_argument('--wd', default=0.0005, type=float)
    parser.add_argument('--save_every', default=5, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument(
        '--path_dir', default='../logs/ssl_framework/', type=str)

    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--name', default=None, type=str)

    args = parser.parse_args()

    if args.submit:
        if args.name:
            name = args.name + '_'
        else:
            name = f'{args.level}'

        print('Submitting the job.')
        os.makedirs(
            './scripts/ssl_framework/sevir/pretraining/augmentations', exist_ok=True)
        os.makedirs(
            './logs/ssl_framework/sevir/pretraining/augmentations', exist_ok=True)
        preamble = (
            f'#!/bin/sh\n#SBATCH --gres=gpu:volta:1\n#SBATCH --cpus-per-task=40\n#SBATCH'
            f'-o ./logs/ssl_framework/sevir/pretraining/augmentations/{name}.out\n#SBATCH '
            f'--job-name=scl_{name}_rumen --mail-user=rumenrd@mit.edu --mail-type=ALL\n\n'
        )
        with open(f'./scripts/ssl_framework/sevir/pretraining/augmentations/{name}.sh', 'w') as file:
            file.write(preamble)
            file.write(
                f'python experiments/ssl_framework/sevir/pretraining/augmentations.py --level={args.level} '
                f'--path_dir=../output/ssl_framework/sevir/pretraining/augmentations/{name}'
            )
            file.write('\n')
        os.system(
            f'sbatch ./scripts/ssl_framework/sevir/pretraining/augmentations/{name}.sh')
    else:
        main(args)
