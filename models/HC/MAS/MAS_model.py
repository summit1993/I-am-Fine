# -*- coding: UTF-8 -*-
import torch.nn as nn

from backbone.backbone import *
from utilities.common_tools import unfreeze_backbone

class MASModel(nn.Module):
    def __init__(self, backbone_name, hierarchy, backbone_unfreeze_layers='all'):
        super(MASModel, self).__init__()
        self.backbone = Backbone[backbone_name](needs_flat=False)
        unfreeze_backbone(self.backbone, backbone_unfreeze_layers)