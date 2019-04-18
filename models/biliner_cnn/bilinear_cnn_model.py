# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch.nn.functional as f
import torch

from backbone.backbone import *

class BilinearCNNModel(nn.Module):
    def __init__(self, backbone_name, label_num, inc=2048,  c1=512, c2=64):
        super(BilinearCNNModel, self).__init__()
        self.backbone = Backbone[backbone_name](needs_flat=False)
        self.conv1 = nn.Conv2d(inc, c1, kernel_size=1)
        self.conv2 = nn.Conv2d(inc, c2, kernel_size=1)
        self.fc = nn.Linear(c1 * c2, label_num)

    def forward(self, x):
        x = self.backbone(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = x1.view(x1.shape[0], x1.shape[1], 1, -1)
        x2 = x2.view(x2.shape[0], 1, x2.shape[1], -1)
        x_b = x1 * x2
        x_b = x_b.sum(-1) / x_b.shape[-1]
        x_b = x_b.view(x_b.shape[0], -1)
        x_b = torch.sign(x_b) * torch.sqrt(torch.abs(x_b))
        x_b = f.normalize(x_b)
        x_out = self.fc(x_b)
        return x_out



