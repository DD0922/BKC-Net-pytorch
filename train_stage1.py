# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 00:24:06 2020

@author: 1
"""

import os
from os.path import join
import argparse
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from model.ConAttUnet import ConAttUNet
from utils.dataloader_seg import RCC3D
from utils.loss import crossentropy, dice_coef, B_crossentropy

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(net_S, opt_S, loss_Seg, dice_S, dataloader_R, epoch, n_epochs, Iters, Savedir):
    segloss_S_log = AverageMeter()
    dice_log = AverageMeter()

    net_S.train()
    for batch_index, (patient, image, gt) in enumerate(dataloader_R):

        opt_S.zero_grad()
        image = image.cuda().float()
        gt = gt.cuda().float()

        seg = net_S(image)
        seg_errS = loss_Seg(seg, gt)
        errS = seg_errS
        errS.backward()
        opt_S.step()

        segloss_S_log.update(seg_errS.item(), n=args.b)

        res = '\t'.join(['Epoch: [%d/%d]' % (epoch, n_epochs),
                         'Iter: [%d/%d]' % (batch_index + 1, Iters),
                         'Loss: %f' % errS.item(),
                         ])
        print(res)
        f = open(os.path.join(Savedir, 'trainiter.txt'), "a+")
        f.write(res+'\n')
        f.close()
        del errS, seg_errS, image, gt

    res = '\t'.join(['Epoch: [%s]' % str(epoch),
                     'SegLoss_%f' % segloss_S_log.sum,
                     'Dice_%f' % dice_log.avg,
                     ])
    print(res)
    f = open(os.path.join(Savedir, 'train.txt'), "a+")
    f.write(res+'\n')
    f.close()
    return

def predict(net_S, loss_Seg, dice_S, dataloader_R, epoch, n_epochs, Savedir):
    print("\nPredict........")
    net_S.eval()
    dice_log = AverageMeter()

    for batch_index, (patient, image, gt) in enumerate(dataloader_R):
        image = image.cuda().float()
        gt = gt.cuda().float()

        seg = net_S(image)
        seg_errS = loss_Seg(seg, gt)
        errS = seg_errS
        
        dice = dice_S(seg[0][0].cpu().detach().numpy(), gt[0][0].cpu().detach().numpy())
        dice_log.update(dice, n=1)
        
        res = '\t'.join(['Valid Epoch: [%d]' % (epoch),
                         'Iter: [%d]' % (batch_index + 1),
                         'Loss: %f' % errS.item(),
                         'Dice_ %f' % dice,
                         'Patient: %s' % patient])
        print(res)
        f = open(os.path.join(Savedir, 'validiter.txt'), "a+")
        f.write(res+'\n')
        f.close()

    res = '\t'.join(['Valid Epoch: [%s]' % str(epoch),
                     'Dice_%f' % dice_log.avg])
    print(res)
    print()
    f = open(os.path.join(Savedir, 'valid.txt'), "a+")
    f.write(res+'\n')
    f.close()
    return dice_log.avg, dice_log.avg


def train_net(n_epochs=800, batch_size=1, lr=1e-4, model_name='demo'):
    save_dir = 'results/' + model_name
    checkpoint_dir = 'weights/' + model_name
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    net_S = ConAttUNet(n_channels=1, n_classes=1)

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net_S.parameters())))
    if torch.cuda.is_available():
        net_S = net_S.cuda()

    train_dataset = RCC3D(5, args.kfold, True)
    valid_dataset = RCC3D(5, args.kfold, False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    
    opt_S = torch.optim.Adam(net_S.parameters(), lr=lr)

    loss_Seg = B_crossentropy()
    Iters = len(train_dataset)
    Dice_S = dice_coef()

    aa = 0.0
    bb = 0.0
    a_b = 0.0
    b_a = 0.0
    for epoch in range(0, n_epochs):  # n_epochs
        train_epoch(net_S, opt_S, loss_Seg, Dice_S, train_dataloader, epoch, n_epochs, Iters,
                    save_dir)
        with torch.no_grad():
            a, b = predict(net_S, loss_Seg, Dice_S, valid_dataloader, epoch, n_epochs, save_dir)
            if a > aa:
                aa = a
                a_b = b
                torch.save(net_S.state_dict(),
                           '{0}/{1}_epoch_{2}_best_a.pth'.format(checkpoint_dir, model_name, epoch))  # _use
            if b > bb:
                bb = b
                b_a = a
                torch.save(net_S.state_dict(), '{0}/{1}_epoch_{2}_best_b.pth'.format(checkpoint_dir, model_name, epoch))
            if a == aa and b > a_b:
                a_b = b
                torch.save(net_S.state_dict(),
                           '{0}/{1}_epoch_{2}_better_ab.pth'.format(checkpoint_dir, model_name, epoch))
            if b == bb and a > b_a:
                b_a = a
                torch.save(net_S.state_dict(),
                           '{0}/{1}_epoch_{2}_better_ba.pth'.format(checkpoint_dir, model_name, epoch))
            elif epoch % 50 == 0:
                torch.save(net_S.state_dict(),
                           '{0}/{1}_epoch_{2}_regular.pth'.format(checkpoint_dir, model_name, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-kfold', type=int, default=-1, help='the k-th fold when cross validation')
    parser.add_argument('-gpunum',type=int,default=0, help='use the kth gpu')
    parser.add_argument('-net', type=str, default='SelfCheckUnet_stage1_', help='network type')
    parser.add_argument('-weight', type=str, help='network weight')
    args = parser.parse_args()
#    torch.backends.cudnn.enabled = False
    train_net(model_name = args.net + '_kfold' + str(args.kfold) + '_batch' + str(args.b))
    