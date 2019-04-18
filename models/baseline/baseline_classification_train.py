# -*- coding: UTF-8 -*-
import torch.optim as optim
from baseline.baseline_classification_model import *
from utilities.data_loader import *
import os

from utilities.model_fn import model_process

def baseline_classification_train(data_set_info_dict, config_info, results_save_dir, model_save_dir):
    data_loaders = get_loaders(data_set_info_dict, config_info)
    model = BaselineClassificationModel(config_info['backbone_name'], data_set_info_dict['label_num'])
    device = config_info['device']
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config_info['lr'])
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)
    log_file_name = os.path.join(results_save_dir, 'baseline_classification_results')
    model_process(model, data_loaders, optimizer, config_info, log_file_name, model_save_dir)
