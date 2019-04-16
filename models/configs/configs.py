# -*- coding: UTF-8 -*-
import torchvision.transforms as transforms
import os

EPOCH_NUM = 50
BATCH_SIZE = 4
BACKBONE_NAME = 'resnet-34'
SHOW_ITERS = 10
MODEL_SAVE_EPOCH = 10

TOTAL_FOLDS = 5

THETA = 2.0
POKER_THETAS = [0.5, 2.0]
GAMMA = 2.0

img_size=224
images_states = {}
images_states['morph'] = {}
images_states['morph']['mean'] = [0.56725002, 0.4921832 , 0.46915039]
images_states['morph']['std'] = [0.0664228 , 0.05526468, 0.05015877]
images_states['chaLearn'] = {}
images_states['chaLearn']['mean'] = [0.56036083, 0.42759175, 0.37782857]
images_states['chaLearn']['std'] = [0.07101307, 0.05536717, 0.0523543 ]

def get_dataset_info(data_set_name):
    info_dict = {}
    if data_set_name == 'morph':
        info_dict['name'] = 'morph'
        info_dict['label_num'] = 62
        info_dict['begin_age'] = 16
        info_dict['min_age'] = 16
        info_dict['max_age'] = 77
        # root_dir = 'D:\\program\\deep_learning\\Deep-HC\\Deep-HC\\data\\morph_50000\\'
        root_dir = '/home1/xcd/program/Deep-HC/data/morph_50000'
        info_dict['root_dir'] = root_dir
        info_dict['info_file'] = os.path.join(root_dir, 'morph_50000_info.txt')
        # info_dict['hierarchy_file'] = os.path.join(root_dir, 'morph_hierarchy.txt')
        info_dict['image_dir'] = os.path.join(root_dir, 'morph_50000_image')
        info_dict['index_file'] = os.path.join(root_dir, 'samples_index.pkl')
        info_dict['split_index_file'] = os.path.join(root_dir, 'split_index_dict.pkl')
        # info_dict['HLDL'] = os.path.join(root_dir, 'morph_HLDL.pkl')
    elif data_set_name == 'chaLearn':
        info_dict['name'] = 'chaLearn'
        info_dict['label_num'] = 89
        info_dict['begin_age'] = 1
        info_dict['min_age'] = 1
        info_dict['max_age'] = 89
        root_dir = 'D:\\program\\deep_learning\\Deep-HC\\Deep-HC\\data\\chaLearn\\'
        # root_dir = '/home1/xcd/program/Deep-HC/data/chaLearn'
        info_dict['root_dir'] = root_dir
        info_dict['info_file'] = os.path.join(root_dir, 'chaLearn_info.txt')
        # info_dict['hierarchy_file'] = os.path.join(root_dir, 'chaLearn_hierarchy.txt')
        info_dict['image_dir'] = os.path.join(root_dir, 'chaLearn_16_image')
        info_dict['split_index_file'] = os.path.join(root_dir, 'split_index_dict.pkl')
        # info_dict['HLDL'] = os.path.join(root_dir, 'chaLearn_HLDL_theta.pkl')
    return info_dict


def get_transform_train(dataset_name):
    images_state = images_states[dataset_name]
    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),  # 先四周填充0，再把图像随机裁剪成img_size*img_size
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(std=images_state['std'], mean=images_state['mean']),
    ])
    return transform_train

def get_transform_test(dataset_name):
    images_state = images_states[dataset_name]
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(std=images_state['std'], mean=images_state['mean']),
    ])
    return transform_test