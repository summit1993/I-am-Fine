# -*- coding: UTF-8 -*-
import torch.nn as nn

from backbone.backbone import *

class BilinearCNNModel(nn.Module):
    def __init__(self, backbone_name, label_num):
        super(BilinearCNNModel, self).__init__()
