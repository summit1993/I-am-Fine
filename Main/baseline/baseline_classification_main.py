# -*- coding: UTF-8 -*-
from baseline.baseline_classification_train import *
from configs.configs import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

configs_class = Configs()
configs_dict = configs_class.configs_dict
data_set_dict = configs_class.data_set_dict

# configs_dict['num_workers'] = 0
# configs_dict['batch_size'] = 4

baseline_classification_train(data_set_dict, configs_dict, 'results', 'models')