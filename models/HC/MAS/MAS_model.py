# -*- coding: UTF-8 -*-
import torch.nn as nn
from functools import partial

from HC.local_model.local_model import get_local_model
from HC.HC_prediction import *
from backbone.backbone import *
from utilities.common_tools import unfreeze_backbone
from HC.MAS.MAS_loss import *

class MASModel(nn.Module):
    # now only support softmax local model
    def __init__(self, backbone_name, hierarchy, backbone_unfreeze_layers='all', local_model_name='softmax', use_all=True):
        super(MASModel, self).__init__()
        self.backbone = Backbone[backbone_name]()
        self.hierarchy  = hierarchy
        self.HC_loss = partial(MAS_loss,  hierarchy=self.hierarchy, use_all=use_all)
        self.HC_prediction = partial(HC_prediction, hierarchy=self.hierarchy, fn='BCE')
        unfreeze_backbone(self.backbone, backbone_unfreeze_layers)
        self.inners_code_list = self.hierarchy['inners_code_list']
        self.heads = nn.Sequential()
        self.heads_index = []
        nodes = self.hierarchy['nodes']
        for code in self.inners_code_list:
            node = nodes[code]
            children_count = len(node.get_children_code())
            if children_count > 1:
                local_model = get_local_model(local_model_name,
                            input_size=[self.backbone.final_feature_num,], label_num=children_count)
                self.heads.add_module(str(code), local_model)
                self.heads_index.append(code)

    def forward(self, x):
        outputs = {}
        x = self.backbone(x)
        for code in self.heads_index:
            outputs[code] = self.heads.__getattr__(str(code))(x)
        return outputs