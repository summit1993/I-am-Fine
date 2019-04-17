# -*- coding: UTF-8 -*-
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, image_list, labels, image_dir, transform=None):
        self.transform = transform
        self.image_dir = image_dir
        self.labels = labels
        self.image_list = image_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.image_list[item])
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is not None:
            label = self.labels[item]
            return img, label
        else:
            return img

def get_loaders(data_dict, config_info):
    loader_dict = {}
    for key in ['train', 'val', 'test']:
        if key not in data_dict:
            continue
        value = data_dict[key]
        loader_set = MyDataset(value['images'], value['labels'], value['image_dir'], value['transform'])
        loader_dict[key] = DataLoader(loader_set, batch_size=config_info['batch_size'], shuffle=value['shuffle'], num_workers=config_info['num_workers'])
    return loader_dict
