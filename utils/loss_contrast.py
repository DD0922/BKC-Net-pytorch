import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Similar to supervised contrastive loss when combining inter and intra- loss.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # print('batch_size', batch_size)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)

            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.t()).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #[batch, view, 128] - > [batch*view, 128]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # print('anchor_feature', anchor_feature.shape)
        # print('contrast_feature', contrast_feature.shape)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.t()), #[6, 128] * [6, 128].t() - > [6, 6]
            self.temperature)
        # print('anchor_dot_contrast', anchor_dot_contrast)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) #[6, 1] 最大的数值
        # print('logits_max', logits_max, logits_max.shape)
        logits = anchor_dot_contrast - logits_max.detach()
        # print('logits', logits)
        # print('mask', mask)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print('mask', mask)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print('logits_mask', logits_mask) # 对角为0的全1矩阵
        mask = mask * logits_mask
        # print('mask2', mask)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  #除了自己都要比
        # print('exp_logits', exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print('log_prob', log_prob)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupConLossOne(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLossOne, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        # print('input fea', features.shape, features)
        # print('input label', labels.shape, labels)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # print('batch_size', batch_size)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)

            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.t()).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # print('constrast_count', contrast_count)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #[batch, view, 128] - > [batch*view, 128]
        # print('contrast_fea', contrast_feature.shape,  contrast_feature)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # print('anchor_feature', anchor_feature.shape, anchor_feature)
        # print('contrast_feature', contrast_feature.shape, contrast_feature)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.t()), #[6, 128] * [6, 128].t() - > [6, 6]
            self.temperature)
        # print('anchor_dot_contrast', anchor_dot_contrast)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) #[6, 1] 最大的数值
        # print('logits_max', logits_max, logits_max.shape)
        logits = anchor_dot_contrast - logits_max.detach()
        # print('logits', logits)
        # print('mask', mask)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print('mask', mask)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print('logits_mask', logits_mask) # 对角为0的全1矩阵
        mask = mask * logits_mask
        # print('mask2', mask)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask #除了自己都要比
        # print('exp_logits', exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print('log_prob', log_prob)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-4)
        # print('mean_log_prob_pos', mean_log_prob_pos)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss