# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import numpy as np
from .logger import create_logger, PD_Stats
import torch.nn as nn
FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

logger = getLogger()

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, shift=2., measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.shift = shift
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = lambda x, y: x.mm(y.t())

        self.max_violation = max_violation
        self.count = 1

    def set_margin(self, margin):
        self.margin = margin

    def loss_func(self, cost, tau):
        cost = (cost - cost.diag().reshape([-1, 1])).exp()
        I = (cost.diag().diag() == 0)
        return cost[I].sum() / (cost.shape[0] * (cost.shape[0] - 1))

    def forward(self, im, s=None, tau=1., lab=None):
        if s is None:
            scores = im
            diagonal = im[:, 0].view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)

            # compare every diagonal score to scores in its column
            # caption retrieval
            cost = (self.margin + scores - d1).clamp(min=0)
            # keep the maximum violating negative for each query
            if self.max_violation:
                cost = cost.max(1)[0]

            return cost.sum()

        else:
            # compute image-sentence score matrix
            scores = self.sim(im, s)
            self.count += 1
            
            diagonal = scores.diag().view(im.size(0), 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)
            mask_s = (scores >= (d1 - self.margin)).float().detach()
            cost_s = scores * mask_s + (1. - mask_s) * (scores - self.shift)
            mask_im = (scores >= (d2 - self.margin)).float().detach()
            cost_im = scores * mask_im + (1. - mask_im) * (scores - self.shift)
            loss = (-cost_s.diag() + tau * (cost_s / tau).exp().sum(1).log() + self.margin).mean() + (-cost_im.diag() + tau * (cost_im / tau).exp().sum(0).log() + self.margin).mean()
            return loss
