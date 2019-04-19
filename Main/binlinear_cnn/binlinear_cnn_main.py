# -*- coding: UTF-8 -*-
from biliner_cnn.bilinear_cnn_model_train import *
from configs.configs import *
import os
from utilities.common_tools import grab_gpu

gpu_id = grab_gpu()
if gpu_id >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

configs_class = Configs()
configs_dict = configs_class.configs_dict
data_set_dict = configs_class.data_set_dict

configs_dict['inc'] = 2048
configs_dict['c1'] = 256
configs_dict['c2'] = 32

configs_dict['batch_size'] = 128
configs_dict['backbone_name'] = 'resnet-50'
configs_dict['show_iters'] = 100
configs_dict['model_save_epoch'] = 1

bilinear_cnn_model_train(data_set_dict, configs_dict, 'results', 'models')