# -*- coding: UTF-8 -*-
import torchvision.transforms as transforms
import pickle
import os
import torch

# root_dir = '/data1/youku'
root_dir = 'D:\\program\\deep_learning\\Deep-HC\\I-am-Fine\\VSR\\data'

class Configs:
    def __init__(self):
        self.configs_dict = {
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            'rfb_num': 4,
            'batch_size': 16,
            'num_workers': 10,
            'lr': 1e-5,
            'weight_decay': 1e-5,
            'epoch_num': 20,
            'show_iters': 1,
            'model_save_epoch':1,
        }
        train_tmp = pickle.load(open(os.path.join(root_dir, 'train.pkl'), 'rb'))
        val_tmp = pickle.load(open(os.path.join(root_dir, 'val.pkl'), 'rb'))
        self.data_set_dict = {
            'train': {'shuffle': True, 'transform': get_transform(),
                      'images': train_tmp, 'volume_k': 2, 'has_hr': True,
                      'image_root_dir': os.path.join(root_dir, 'images')},
            'val': {'shuffle': False, 'transform': get_transform(),
                      'images': val_tmp, 'volume_k': 2, 'has_hr': True,
                      'image_root_dir': os.path.join(root_dir, 'images')},
        }

images_mean = [0.5, 0.5, 0.5]
images_std = [0.5, 0.5, 0.5]
def get_transform():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(std=images_std, mean=images_mean),
        ])
    return transform