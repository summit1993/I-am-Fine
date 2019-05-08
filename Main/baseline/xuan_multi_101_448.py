# -*- coding: UTF-8 -*-
from baseline.baseline_classification_train import *
from configs.configs import *
import os
from utilities.common_tools import grab_gpu

# gpu_id = grab_gpu(frequency=1.5)
# if gpu_id >= 0:
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

os.environ['CUDA_VISIBLE_DEVICES'] = '2,4,5,11,7'

configs_class = Configs(img_size=448)
configs_dict = configs_class.configs_dict
data_set_dict = configs_class.data_set_dict

configs_dict['batch_size'] = 100
configs_dict['backbone_name'] = 'resnet-101'
configs_dict['multi-gpu'] =True

# configs_dict['num_workers'] = 0
# configs_dict['batch_size'] = 4

baseline_classification_train(data_set_dict, configs_dict, 'xuan_results_101_448', 'xuan_models_101_448')