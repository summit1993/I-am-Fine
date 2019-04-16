# -*- coding: UTF-8 -*-
from models.flat_method.classification.baseline.baseline_classification_train import *
from models.configs.configs import *

configs_class = Configs()
configs_dict = configs_class.configs_dict
data_set_dict = configs_class.data_set_dict

baseline_classification_train(data_set_dict, configs_dict, 'results', 'models')