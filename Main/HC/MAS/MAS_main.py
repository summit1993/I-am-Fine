# -*- coding: UTF-8 -*-
from HC.MAS.MAS_train import *
from configs.configs import *
import os
from utilities.common_tools import grab_gpu

gpu_id = grab_gpu()
if gpu_id >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

configs_class = Configs()
configs_dict = configs_class.configs_dict
data_set_dict = configs_class.data_set_dict

configs_dict['batch_size'] = 128
configs_dict['backbone_name'] = 'resnet-101'
configs_dict['backbone_unfreeze_layers'] = ['layer3', 'layer4']
configs_dict['pre_model'] = os.path.join(configs_dict['model_zoo'], 'baseline_fc_101_checkpoint_6.tar')
configs_dict['show_iters'] = 100
configs_dict['model_save_epoch'] = 1
configs_dict['use_all'] = False

MAS_model_train(data_set_dict, configs_dict, 'results', 'models')