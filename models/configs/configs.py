# -*- coding: UTF-8 -*-
import torchvision.transforms as transforms
import pickle
import os
import torch

root_dir = "/home1/CVPR"
program_dir = "/home1/xcd/program/I-am-Fine"
# root_dir = "C:\\Users\\summit\\Desktop\\CVPR_Workshop\\data"

class Configs:
    def __init__(self):
        configs_dict = {
            'device':  torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'epoch_num': 20,
            'show_iters': 10,
            'model_save_epoch': 1,
            'backbone_name': 'resnet-101',
            'batch_size': 256,
            'num_workers': 10,
            'lr': 1e-5,
            'backbone_unfreeze_layers': 'all',
            'pre_model': None,
            'model_zoo': os.path.join(program_dir, 'model_zoo'),
            'model_type': 'baseline',
        }
        self.configs_dict = configs_dict
        train_tmp = pickle.load(open(os.path.join(root_dir, 'data/train.pkl'), 'rb'))
        val_tmp = pickle.load(open(os.path.join(root_dir, 'data/val.pkl'), 'rb'))
        test_tmp = pickle.load(open(os.path.join(root_dir, 'data/test.pkl'), 'rb'))
        data_set_dict = {
            'label_num': 2019,
            'train': {'shuffle': True, 'transform': get_transform_train(),
                      'images': train_tmp['images'], 'labels': train_tmp['labels'],
                      'image_dir': os.path.join(root_dir, 'images/train')},
            'val': {'shuffle': False, 'transform': get_transform_inference(),
                    'images': val_tmp['images'], 'labels': val_tmp['labels'],
                    'image_dir': os.path.join(root_dir, 'images/val')},
            'test': {'shuffle': False, 'transform': get_transform_inference(),
                     'images': test_tmp['images'], 'labels': None,
                     'image_dir': os.path.join(root_dir, 'images/test')}
        }
        self.data_set_dict = data_set_dict

img_size=224
images_mean = [0.5, 0.5, 0.5]
images_std = [0.5, 0.5, 0.5]

def get_transform_train():
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.RandomCrop((img_size, img_size), padding=4),  # 先四周填充0，再把图像随机裁剪成img_size*img_size
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(std=images_std, mean=images_mean),
    ])
    return transform_train

def get_transform_inference():
    transform_inference = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(std=images_std, mean=images_std),
    ])
    return transform_inference