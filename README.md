# I am Fine

## Envoriment
* Python 3.6 [+]
* Pytorch 1.0
* Ubuntu 16.04
* CUDA 8.0 [+]

## Third Party
* Compact Bilinear Pooling (thanks very much)
 &ensp; https://github.com/gdlg/pytorch_compact_bilinear_pooling

## Config
* add the models module to the system envoriment, e.g., 
&ensp; export PYTHONPATH=$PYTHONPATH:/home1/xcd/program/I-am-Fine/models

## Run
* Run the python file in Main folders, e.g., 
&ensp; python baseline_classification_main.py

## Models
* Bilinear CNN Models for Fine-grained Visual Recognition. CVPR, 2015
* Compact Bilinear Pooling. CVPR, 2016

## More
| Algorithm | CUB-200-2011 (%) | Stanford Cars (%) | Stanford Dogs (%) |  FGVC-Aircraft (%) |
| :------: | :------: | :------: |  :------: | :------: |
| 1 | 87.9 | 94.1 | - | 92.1 |
| 2 | 87.5 | 93.9 | - | 91.4 | 
| 3 | 90.4 |  -  | 97.1 | - | 
| 4 | 85.4 | - | 88.5 |  89.1 |
| 5 | 85.3 | 92.5 | 87.3 | - | 
| 6 | 89.4 | 94.5 | 92.2 | 93.0 |

1. Generating Attention from Classifier Activations for Fine-grained Recognition. arXiv, 2018-11
2. Learning to Navigate for Fine-grained Classification. ECCV, 2018
3. Weakly Supervised Complementary Parts Models for Fine-Grained Image Classification from the Bottom Up. arXiv, 2019-3
4. Guided Zoom: Questioning Network Evidence for Fine-grained Classification. arXiv, 2018-12
5. Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-grained Image Recognition. CVPR, 2017
6. See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification. arXiv, 2019-3
