# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch.nn.functional as f
import torch
from backbone.backbone import *
from utilities.common_tools import unfreeze_backbone
from compact_bilinear_pooling import CompactBilinearPooling

class CompactBilinearCNNModel(nn.Module):
    # 6024: her birthday
    def __init__(self, backbone_name, label_num, inc=2048, outc=6024, backbone_unfreeze_layers='all'):
        super(CompactBilinearCNNModel, self).__init__()
        self.backbone = Backbone[backbone_name](needs_flat=False)
        unfreeze_backbone(self.backbone, backbone_unfreeze_layers)
        self.mcb =  CompactBilinearPooling(inc, inc, outc)
        self.c_bilinear_fc = nn.Linear(outc, label_num)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = self.mcb(x, x)
        x = x.sum(1)
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-5)
        x = f.normalize(x)
        x = self.c_bilinear_fc(x)
        return x
