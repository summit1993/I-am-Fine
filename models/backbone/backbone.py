# -*- coding: UTF-8 -*-
from models.backbone.resnet import *

Backbone = {'resnet-18': lambda needs_flat=True: resnet18(pretrained=True, needs_flat=needs_flat),
        'resnet-34': lambda needs_flat=True: resnet34(pretrained=True, needs_flat=needs_flat),
        'resnet-50': lambda needs_flat=True: resnet50(pretrained=True, needs_flat=needs_flat),
        'resnet-101': lambda needs_flat=True: resnet101(pretrained=True, needs_flat=needs_flat),
        'resnet-152': lambda needs_flat=True: resnet152(pretrained=True, needs_flat=needs_flat)}

if __name__ == '__main__':
    back_bone = Backbone['resnet-34'](False)
    print(back_bone.final_feature_num)
#     for parameter in back_bone.parameters():
#             print(parameter.size())
#     extract_feature_num =back_bone.final_feature_num
    # fc = nn.Linear(backbone.final_feature_num, num_classes)
    # self.model.add_module('fc', fc)