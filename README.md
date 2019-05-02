# I am Fine

## 环境
* Python 3.6 [+]
* Pytorch 1.0
* Ubuntu 16.04
* CUDA 8.0 [+]

## 第三方
* Compact Bilinear Pooling (thanks very much)

https://github.com/gdlg/pytorch_compact_bilinear_pooling

## 配置
在环境变量下加入models模块，例如
export PYTHONPATH=$PYTHONPATH:/home1/xcd/program/I-am-Fine/models

## 运行
运行Main目录下的python文件，如python baseline_classification_main.py

## Models
* Bilinear CNN Models for Fine-grained Visual Recognition (CVPR, 15)
* Diversified Visual Attention Networks for Fine-Grained Object Classification (IEEE Trans. Multimedia, 17)
* Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition (CVPR, 17)

## More
|  Algorithm | CUB-200-2011 (%) | Stanford Cars (%) | FGVC-Aircraft (%) |  Stanford Dogs (%) |
| :------: | :------: | :------: |  :------: | :------: |
| 1 | 87.9 | 94.1 | 92.1 | - |
| 2 | 87.5 | 93.9 | 91.4 | - | 
| 3 | 90.4 |  -  | - | 97.1 | 
| 4 | 85.4 | - | 89.1 |  88.5 |


1. Generating Attention from Classifier Activations for Fine-grained Recognition, arXiv, 2018-11
2. Learning to Navigate for Fine-grained Classification, ECCV, 2018
3. Weakly Supervised Complementary Parts Models for Fine-Grained Image Classification from the Bottom Up, arXiv, 2019-3
4. Guided Zoom: Questioning Network Evidence for Fine-grained Classification, arXiv, 2018-12




