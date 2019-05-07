import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE

def train_transform(img):
    # if len(img.shape) == 2:
    #     img = np.stack([img] * 3, 2)
    # img = Image.fromarray(img, mode='RGB')
    img = transforms.Resize(INPUT_SIZE)(img)
    # img = transforms.Resize((600, 600), Image.BILINEAR)(img)
    # img = transforms.RandomCrop(INPUT_SIZE)(img)
    img = transforms.RandomHorizontalFlip()(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.5, 5, 5], [0.5, 0.5, 0.5])(img)
    return img

def inference_transform(img):
    # if len(img.shape) == 2:
    #     img = np.stack([img] * 3, 2)
    # img = Image.fromarray(img, mode='RGB')
    img = transforms.Resize(INPUT_SIZE)(img)
    # img = transforms.Resize((600, 600), Image.BILINEAR)(img)
    # img = transforms.CenterCrop(INPUT_SIZE)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
    return img


class MyDataset():
    def __init__(self, image_list, labels, image_dir, transform_mode='train'):
        if transform_mode == 'train':
            self.transform = train_transform
        else:
            self.transform = inference_transform
        self.image_dir = image_dir
        self.labels = labels
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.image_list[item])
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        if self.labels is not None:
            label = self.labels[item]
            return img, label
        else:
            return img