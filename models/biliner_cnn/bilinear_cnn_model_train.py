# -*- coding: UTF-8 -*-
import torch.optim as optim
from biliner_cnn.bilinear_cnn_model import *
from utilities.data_loader import *
import os
from utilities.model_fn import model_process

def bilinear_cnn_model_train(data_set_info_dict, config_info, results_save_dir, model_save_dir):
    data_loaders = get_loaders(data_set_info_dict, config_info)
    model = BilinearCNNModel(config_info['backbone_name'], data_set_info_dict['label_num'],
                             config_info['inc'], config_info['c1'], config_info['c2'],
                             config_info['backbone_unfreeze_layers'])
    if config_info['pre_model'] is not None:
        checkpoint = torch.load(config_info['pre_model'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.train()
    device = config_info['device']
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config_info['lr'], weight_decay=config_info['weight_decay'])
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)
    log_file_name = os.path.join(results_save_dir, 'biliner_model_results')
    model_process(model, data_loaders, optimizer, config_info, log_file_name, model_save_dir)
