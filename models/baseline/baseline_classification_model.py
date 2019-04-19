# -*- coding: UTF-8 -*-
import torch.nn as nn

from backbone.backbone import *

class BaselineClassificationModel(nn.Module):
    def __init__(self, backbone_name, label_num, fine_tune_backbone=True):
        super(BaselineClassificationModel, self).__init__()
        self.backbone = Backbone[backbone_name]()
        if not fine_tune_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.fc = nn.Linear(self.backbone.final_feature_num, label_num)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x