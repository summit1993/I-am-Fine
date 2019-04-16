# -*- coding: UTF-8 -*-
import torch.optim as optim

from flat_method.classification.baseline_classification import *
from utilities.data_loader import *
import os

from utilities.model_fn import model_process

def baseline_classification(data_set_info_dict, config_info, device, results_save_dir, model_save_dir):
    data_loaders = get_loaders(data_set_info_dict, config_info)
    model = BaselineClassificationModel(config_info['backbone_name'], data_set_info_dict['label_num'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)
    log_file_name = os.path.join(results_save_dir, 'baseline_classification_results.pkl')
    model_process(model, data_loaders, optimizer, config_info, log_file_name, model_save_dir)


