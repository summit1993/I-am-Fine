# -*- coding: UTF-8 -*-
import torch.nn as nn
from backbone.backbone import *
from utilities.common_tools import unfreeze_backbone
import queue
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

inner_channel = 128

def create_local_module(input_channel, output_channel, feature_num, label_num):
    local_module = nn.Sequential()
    conv1 = nn.Conv2d(input_channel, inner_channel, kernel_size=1, padding=0)
    bn1 = nn.BatchNorm2d(inner_channel)
    conv2 = nn.Conv2d(inner_channel, inner_channel, kernel_size=3, padding=1)
    bn2 = nn.BatchNorm2d(inner_channel)
    conv3 = nn.Conv2d(inner_channel, output_channel, kernel_size=1, padding=0)
    bn3 = nn.BatchNorm2d(output_channel)
    local_extract = nn.Sequential()
    local_extract.add_module('conv1', conv1)
    local_extract.add_module('bn1', bn1)
    local_extract.add_module('relu1', nn.ReLU())
    local_extract.add_module('conv2', conv2)
    local_extract.add_module('bn2', bn2)
    local_extract.add_module('relu2', nn.ReLU())
    local_extract.add_module('conv3', conv3)
    local_extract.add_module('bn3', bn3)
    local_module.add_module('local_extract', local_extract)
    local_module.add_module('pool', nn.AdaptiveAvgPool2d(1))
    fc = nn.Linear(feature_num, label_num)
    local_module.add_module('fc', fc)
    return local_module

class PokerModel(nn.Module):
    def __init__(self, backbone_name, hierarchy, backbone_unfreeze_layers='all', ):
        super(PokerModel, self).__init__()
        input_channel = in_channels_dict[backbone_name]
        self.backbone = Backbone[backbone_name](False)
        unfreeze_backbone(self.backbone, backbone_unfreeze_layers)
        inner_feature_num = input_channel * self.backbone.block_expansion
        self.hierarchy = hierarchy
        self.HC_loss = partial(Poker_loss, hierarchy=self.hierarchy)
        self.HC_prediction = partial(HC_prediction, hierarchy=self.hierarchy)
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
                local_input_channel = input_channel
            else:
                label_num = children_count + 1
                local_input_channel = input_channel
            local_module = create_local_module(local_input_channel, input_channel,
                                               inner_feature_num, label_num)
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
                in_x = x
            else:
                parent_x = feature_outputs_dict[node.get_parent_code()]
                # in_x = torch.cat([parent_x, x], dim=1)
                in_x = parent_x + x
            in_x = self.relu(in_x)
            local_feature = local_module.local_extract(in_x)
            feature_outputs_dict[code] = local_feature
            local_x = local_module.pool(local_feature)
            local_x = local_x.view(local_x.size(0), -1)
            local_y = local_module.fc(local_x)
            prediction_outputs_dict[code] = local_y
        if return_final_features:
            return prediction_outputs_dict, feature_outputs_dict
        else:
            return prediction_outputs_dict