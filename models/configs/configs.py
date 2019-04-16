# -*- coding: UTF-8 -*-
import torchvision.transforms as transforms

class Configs:
    def __init__(self):
        configs_dict = {}
        configs_dict['device'] = 'CPU'
        configs_dict['epoch_num'] = 50
        configs_dict['show_iters'] = 10
        configs_dict['model_save_epoch'] = 5
        configs_dict['backbone_name'] = 'resnet-101'
        configs_dict['batch_size'] = 256
        self.configs_dict = configs_dict

class DataSetInfo:
    def __init__(self):
        data_set_dict = {
            'train': {'shuffle': True, 'transform': get_transform_train()},
            'val': {'shuffle': False, 'transform': get_transform_inference()},
            'test': {'shuffle': False, 'transform': get_transform_inference()}
        }
        self.data_set_dict = data_set_dict

img_size=224
images_mean = [0.5, 0.5, 0.5]
images_std = [0.5, 0.5, 0.5]

def get_transform_train():
    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),  # 先四周填充0，再把图像随机裁剪成img_size*img_size
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(std=images_std, mean=images_mean),
    ])
    return transform_train

def get_transform_inference():
    transform_inference = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(std=images_std, mean=images_std),
    ])
    return transform_inference