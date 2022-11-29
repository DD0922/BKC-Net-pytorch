# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 00:22:19 2020

@author: 1
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
from torch.nn.functional import max_pool3d

class dice_coef(nn.Module):
    def __init__(self):
        super(dice_coef, self).__init__()

    def forward(self, pred, gt):
        smooth = 1e-5

        pred = np.where(pred >= 0.5, 1, 0)
        gt = np.where(gt == 1, 1, 0)

        a = np.sum(pred * gt)
        b = np.sum(pred)
        c = np.sum(gt)

        dice = (2 * a + smooth) / (b + c + smooth)
        return dice

import torch
import torch.nn as nn
class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 获取每个批次的大小 N
        N = targets.size()[0]
        # 平滑变量
        smooth = 1e-4
        # 将宽高 reshape 到同一纬度
        input_flat = input.contiguous().view(N, -1)
        targets_flat = targets.contiguous().view(N, -1)

        # 计算交集
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
        return loss

class B_crossentropy(nn.Module):
    def __init__(self):
        super(B_crossentropy, self).__init__()

    def forward(self, y_pred, y_true):
        smooth = 1e-4
        # print('BCE', y_pred, y_true)
        return -torch.mean(y_true * torch.log(y_pred+smooth)+(1-y_true)*torch.log(1-y_pred+smooth))

class weightedFL(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super(weightedFL, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.wm = 74 / 174.0
        self.wn = 100 / 174.0

    def forward(self, _input, target):
        # target=0 high
        # target=1 low
        pt = _input

        """-torch.mean(y_true * torch.log(y_pred+smooth)+(1-y_true)*torch.log(1-y_pred+smooth)) """
        loss = - self.wm * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               self.wn * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss