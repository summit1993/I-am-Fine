# -*- coding: UTF-8 -*-
from FSTRN_train import *
from FSTRN_config import *

configs_class = Configs()
configs_dict = configs_class.configs_dict
data_set_dict = configs_class.data_set_dict

FSTRN_train(data_set_dict, configs_dict, 'results', 'models')