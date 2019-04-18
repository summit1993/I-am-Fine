import os
import numpy as np
from PIL import Image
import pynvml
import time

def calculate_rgb_mean_and_std(img_dir_list, img_size=224):
    img_mean = np.zeros(3)
    img_std = np.zeros(3)
    img_count = 0.0

    for img_dir in img_dir_list:
        img_list = os.listdir(img_dir)
        img_count += len(img_list)
        for img_name in img_list:
            img_path = os.path.join(img_dir, img_name)
            # img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (img_size, img_size))
            img = Image.open(img_path)
            img = img.resize((img_size, img_size))
            img = np.array(img, dtype=np.float32)
            img = img / 255.0
            for i in range(3):
                img_mean[i] = img_mean[i] + img[:, :, i].mean()
    img_mean = img_mean / img_count

    for img_dir in img_dir_list:
        img_list = os.listdir(img_dir)
        for img_name in img_list:
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path)
            img = img.resize((img_size, img_size))
            img = np.array(img, dtype=np.float32)
            img = img / 255.0
            for i in range(3):
                img_std[i] = img_std[i] + ((img[:, :, i] - img_mean[i]) ** 2).sum()
    img_std = img_std / (img_count * img_size * img_size)
    return img_mean, img_std


def grab_gpu(frequency=1, total=3600 * 24, gpu_mem=9):
    for t in range(total):
        max_value, max_index = -1, -1
        for i in range(12):
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem = meminfo.free / 1024 / 1024 / 1024
            if free_mem > max_value:
                max_value = free_mem
                max_index = i
        # print(max_value, max_index)
        if max_value >= gpu_mem:
            print(max_value, max_index)
            return max_index
        time.sleep(frequency)
    return -1