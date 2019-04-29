# -*- coding: UTF-8 -*-
import torch.nn as nn

from backbone.backbone import *
from utilities.common_tools import unfreeze_backbone

class BaselineClassificationModel(nn.Module):
    def __init__(self, backbone_name, label_num, backbone_unfreeze_layers='all'):
        super(BaselineClassificationModel, self).__init__()
        self.backbone = Backbone[backbone_name]()
        unfreeze_backbone(self.backbone, backbone_unfreeze_layers)
        self.fc = nn.Linear(self.backbone.final_feature_num, label_num)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x