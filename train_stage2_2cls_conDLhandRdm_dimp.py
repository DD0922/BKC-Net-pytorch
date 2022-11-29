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
from torch.optim.lr_scheduler import StepLR

from model.Contrast_2cls_rdm_dl_dimp import OneStageContrastDLRdm1
from utils.dataloader_2cls_conDLTexturehandRdm import CCRCC3D
from utils.loss_contrast import SupConLoss

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
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class CLSCriterion(object):
    def __init__(self, clsnum):
        self.num = clsnum
        self.reset()

    def reset(self):
        self.acc = 0
        self.conf_mat = np.zeros([self.num, self.num])
        self.auc = 0
        self.predlabels = []
        self.auclabel = []

    def update(self, pred, labels):
        labels = labels.cpu().data.item()
        self.auclabel.append(labels)
        cate_i = labels
        pre_i = pred
        if cate_i == pre_i:
            self.acc += 1
        self.conf_mat[int(cate_i), int(pre_i)] += 1.0
        self.predlabels.append(pre_i)

    def cal(self, epoch, Savedir):
        print("Conf mat:")
        f = open(os.path.join(Savedir, 'res.txt'), "a+")
        f.write('Epoch=%d\n' % epoch)
        print("Conf mat:")
        for i in range(0, self.num):
            for j in range(0, self.num):
                print(self.conf_mat[i][j], end = ' ')
                f.write('%d ' % self.conf_mat[i][j])
            print()
            f.write( '\n')
        f.close()
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
        self.acc = self.acc * 1.0 / self.conf_mat.sum()
        self.precision = precision_score(self.auclabel, self.predlabels)
        self.recall = recall_score(self.auclabel, self.predlabels)
        self.f1 = f1_score(self.auclabel, self.predlabels)
        print('ACC_%.4f' % self.acc,
              'Precision_%.4f' % self.precision,
              'Recall_%.4f' % self.recall,
              'F1_%.4f' % self.f1)

def train_epoch(net_S, opt_S, loss_Cls, dataloader_R, epoch, n_epochs, Iters, Savedir):

    net_S.train()
    contrastloss = SupConLoss()
    contrastloss_S_log = AverageMeter()
    X_train = []
    Y_train = []
    for batch_index, (patient, dlfea, rdmfea, ccrcc_cls) in enumerate(dataloader_R):
        opt_S.zero_grad()
        Y_train.extend(ccrcc_cls.cpu().numpy().tolist())
        dlfea = dlfea.cuda()
        rdmfea = rdmfea.cuda()
        ccrcc_cls = ccrcc_cls.cuda()

        dl_x, rdm_x, cls2 = net_S(dlfea, rdmfea)
        features = torch.cat([dl_x.unsqueeze(1), rdm_x.unsqueeze(1)], dim=1)
        X_train.extend(features.view(features.shape[0], -1).cpu().data.numpy().tolist())
        contrast_errS = contrastloss(features, ccrcc_cls)
        errS = contrast_errS
        errS.backward()
        opt_S.step()

        contrastloss_S_log.update(contrast_errS.item(), n=ccrcc_cls.size(0))

        res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                         'Iter: [%d/%d]' % (batch_index + 1, Iters),
                         'Loss: %f' % errS.item(),
                         'ContrastLoss: %f' % contrast_errS.item(),
                         ])
        print(res)
        f = open(os.path.join(Savedir, 'trainiter.txt'), "a+")
        f.write(res + '\n')
        f.close()

    res = '\t'.join(['Epoch: [%s]' % str(epoch + 1),
                     'ContrastLoss: %f' % contrastloss_S_log.sum,
                     ])
    print(res)
    f = open(os.path.join(Savedir, 'train.txt'), "a+")
    f.write(res + '\n')
    f.close()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    return X_train, Y_train

def predict(net_S, loss_Cls, dataloader_R, epoch, Savedir):
    print("\nPredict........")
    print()
    net_S.eval()

    cls2loss_S_log = AverageMeter()
    contrastloss_S_log = AverageMeter()
    contrastloss = SupConLoss()
    cls2_log = CLSCriterion(2)


    for batch_index, (patient, dlfea, rdmfea, ccrcc_cls) in enumerate(dataloader_R):

        dlfea = dlfea.cuda()
        rdmfea = rdmfea.cuda()
        ccrcc_cls = ccrcc_cls.cuda()

        dl_x, rdm_x, cls2 = net_S(dlfea, rdmfea)
        features = torch.cat([dl_x.unsqueeze(1), rdm_x.unsqueeze(1)], dim=1)
        X_valid = features.view(1, args.p*2).cpu().data.numpy()
        result = knn.predict(X_valid)
        # print('X_valid', X_valid, result)
        contrast_errS = contrastloss(features, ccrcc_cls)
        errS = contrast_errS

        contrastloss_S_log.update(contrast_errS.item(), n=1)
        cls2_log.update(result[0], ccrcc_cls)

        res = '\t'.join(['Valid Epoch: [%d]' % (epoch + 1),
                         'Iter: [%d]' % (batch_index + 1),
                         'Loss: %f' % errS.item(),
                         'ContrastLoss: %f' % contrast_errS.item(),
                         '2label: %d' % ccrcc_cls.cpu().data,
                         '2Pred: %d' % result[0],
                         'Patient: %s' % patient])
        print(res)
        f = open(os.path.join(Savedir, 'validiter.txt'), "a+")
        f.write(res + '\n')
        f.close()

    cls2_log.cal(epoch, Savedir)

    res = '\t'.join(['Valid Epoch: [%s]' % str(epoch + 1),
                     '2Loss_%f' % cls2loss_S_log.sum,
                     '2Acc_%f' % cls2_log.acc,
                     '2AUC_%f' % cls2_log.auc,
                     'Precision_%f' % cls2_log.precision,
                     'Recall_%f' % cls2_log.recall,
                     'F1_%f' % cls2_log.f1,
                     'ContrastLoss_%f' % contrastloss_S_log.sum])
    print(res)
    print()
    f = open(os.path.join(Savedir, 'valid.txt'), "a+")
    f.write(res + '\n')
    f.close()
    return cls2_log.acc, cls2_log.auc


def train_net(n_epochs=2000, batch_size=32, model_name='demo'):

    save_dir = 'results/' + model_name
    checkpoint_dir = 'weights/' + model_name

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    print('dir:', checkpoint_dir)

    net_S = OneStageContrastDLRdm1(dlfeanum=1920, rdmfeanum=391, dimp=args.p)

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net_S.parameters())))
    if torch.cuda.is_available():
        net_S = net_S.cuda()

    train_dataset = CCRCC3D(5, args.kfold, True)
    valid_dataset = CCRCC3D(5, args.kfold, False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    opt_S = torch.optim.Adam(filter(lambda p: p.requires_grad, net_S.parameters()), lr=args.lr)

    loss_Cls = nn.CrossEntropyLoss()
    Iters = len(train_dataset)

    aa = 0.0
    bb = 0.0
    a_b = 0.0
    b_a = 0.0
    for epoch in range(0, n_epochs):  # n_epochs
        x_train, y_train = train_epoch(net_S, opt_S, loss_Cls, train_dataloader, epoch, n_epochs, Iters, save_dir)
        knn.fit(x_train, y_train)
        with torch.no_grad():
            a, b = predict(net_S, loss_Cls, valid_dataloader, epoch, save_dir)
            if a > aa:
                aa = a
                a_b = b
                joblib.dump(knn, '{0}/{1}_epoch_{2}_best_a.m'.format(checkpoint_dir, model_name, epoch))
                torch.save(net_S.state_dict(),
                           '{0}/{1}_epoch_{2}_best_a.pth'.format(checkpoint_dir, model_name, epoch))  # _use
            if b > bb:
                bb = b
                b_a = a
                joblib.dump(knn, '{0}/{1}_epoch_{2}_best_b.m'.format(checkpoint_dir, model_name, epoch))
                torch.save(net_S.state_dict(), '{0}/{1}_epoch_{2}_best_b.pth'.format(checkpoint_dir, model_name, epoch))
            if a == aa and b > a_b:
                a_b = b
                joblib.dump(knn, '{0}/{1}_epoch_{2}_better_ab.m'.format(checkpoint_dir, model_name, epoch))
                torch.save(net_S.state_dict(),
                           '{0}/{1}_epoch_{2}_better_ab.pth'.format(checkpoint_dir, model_name, epoch))
            if b == bb and a > b_a:
                b_a = a
                joblib.dump(knn, '{0}/{1}_epoch_{2}_better_ba.m'.format(checkpoint_dir, model_name, epoch))
                torch.save(net_S.state_dict(),
                           '{0}/{1}_epoch_{2}_better_ba.pth'.format(checkpoint_dir, model_name, epoch))
            elif epoch % 50 == 0:
                joblib.dump(knn, '{0}/{1}_epoch_{2}_regular.m'.format(checkpoint_dir, model_name, epoch))
                torch.save(net_S.state_dict(),
                           '{0}/{1}_epoch_{2}_regular.pth'.format(checkpoint_dir, model_name, epoch))


if __name__ == '__main__':
    for kfold in range(1, 5):
        parser = argparse.ArgumentParser()
        parser.add_argument('-w', type=int, default=1, help='number of workers for dataloader')
        parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
        parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
        parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
        parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
        parser.add_argument('-kfold', type=int, default=kfold, help='the k-th fold when cross validation')
        parser.add_argument('-gpunum', type=int, default=0, help='use the kth gpu')
        parser.add_argument('-p', type=int, default=128, help='the dimension of the prototype')
        parser.add_argument('-net', type=str, default='BKCNet2_stage2_Adam_2cls_connormDLtextureRdm4_7KNN', help='network type')
        parser.add_argument('-weight', type=str, help='network weight')
        args = parser.parse_args()
        from sklearn.neighbors import KNeighborsClassifier
        import joblib
        knn = KNeighborsClassifier(n_neighbors=7)
        train_net(batch_size=args.b, model_name=args.net + '_kfold' + str(args.kfold) + '_batch' + str(args.b) + '_p' + str(args.p))
