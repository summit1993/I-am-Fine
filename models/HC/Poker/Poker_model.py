# -*- coding: UTF-8 -*-
import torch.nn as nn
from backbone.backbone import *
from utilities.common_tools import unfreeze_backbone
from functools import partial
from HC.Poker.Poker_loss import *
from HC.HC_prediction import *

in_channels_dict = {
    'resnet-18': 512,
    'resnet-34': 512,
    'resnet-50': 2048,
    'resnet-101': 2048,
    'resnet-152': 2048
}

inner_channel = 64
output_channel = 256

def create_local_module(input_channel, label_num):
    local_module = nn.Sequential()
    local_extract = nn.Sequential(
        nn.Conv2d(input_channel, inner_channel, kernel_size=1, padding=0),
        nn.BatchNorm2d(inner_channel),
        nn.ReLU(),
        nn.Conv2d(inner_channel, inner_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(inner_channel),
        nn.ReLU(),
        nn.Conv2d(inner_channel, output_channel, kernel_size=1, padding=0),
        nn.BatchNorm2d(output_channel)
    )
    local_module.add_module('local_extract', local_extract)
    fuse_extract = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(output_channel, output_channel, kernel_size=1, padding=0),
        nn.BatchNorm2d(output_channel)
    )
    local_module.add_module('fuse_extract', fuse_extract)
    local_module.add_module('pool', nn.AdaptiveAvgPool2d(1))
    fc = nn.Linear(output_channel, label_num)
    local_module.add_module('fc', fc)
    return local_module

class PokerModel(nn.Module):
    def __init__(self, backbone_name, hierarchy, backbone_unfreeze_layers='all', loss_fn='softmax'):
        super(PokerModel, self).__init__()
        input_channel = in_channels_dict[backbone_name]
        self.backbone = Backbone[backbone_name](False)
        unfreeze_backbone(self.backbone, backbone_unfreeze_layers)
        self.hierarchy = hierarchy
        self.HC_loss = partial(Poker_loss, hierarchy=self.hierarchy, loss_fn=loss_fn)
        self.HC_prediction = partial(HC_prediction, hierarchy=self.hierarchy, fn=loss_fn)
        self.inners_code_list = self.hierarchy['inners_code_list']
        nodes = self.hierarchy['nodes']
        self.relu = nn.ReLU()
        self.local_modules = nn.Sequential()
        for code in self.inners_code_list:
            node = nodes[code]
            children_count = len(node.get_children_code())
            if children_count <= 1:
                continue
            if code == -1:
                label_num = children_count
            else:
                if loss_fn == 'softmax':
                    label_num = children_count + 1
                else:
                    label_num = children_count
            local_input_channel = input_channel
            local_module = create_local_module(local_input_channel, label_num)
            self.local_modules.add_module(str(code), local_module)

    def forward(self, x, return_final_features=False):
        que = queue.Queue()
        que.put(-1)
        nodes = self.hierarchy['nodes']
        x = self.backbone(x)
        feature_outputs_dict = {}
        prediction_outputs_dict = {}
        while not que.empty():
            code = que.get()
            node = nodes[code]
            children_code = node.get_children_code()
            children_count = len(children_code)
            if children_count == 0:
                continue
            for child_code in children_code:
                que.put(child_code)
            local_module = self.local_modules.__getattr__(str(code))
            if code == -1:
                parent_x = 0.0
            else:
                parent_x = feature_outputs_dict[node.get_parent_code()]
            local_feature = local_module.local_extract(x)
            local_feature = local_module.fuse_extract(local_feature + parent_x)
            feature_outputs_dict[code] = local_feature
            local_x = local_module.pool(local_feature)
            local_x = local_x.view(local_x.size(0), -1)
            local_y = local_module.fc(local_x)
            prediction_outputs_dict[code] = local_y
        if return_final_features:
            return prediction_outputs_dict, feature_outputs_dict
        else:
            return prediction_outputs_dict