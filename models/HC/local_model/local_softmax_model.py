# -*- coding: UTF-8 -*-
import torch.nn as nn

class LocalSoftmaxModel(nn.Module):

    def __init__(self, input_size, label_num):
        # input_size: (d1, d2, ...)
        super(LocalSoftmaxModel, self).__init__()
        self.input_size = input_size
        input_flat = 1
        for d in self.input_size:
            input_flat *= d
        self.local_model_softmax_fc_1 = nn.Linear(input_flat, 1024)
        self.relu = nn.ReLU()
        self.local_model_softmax_fc_2 = nn.Linear(1024, label_num)

    def forward(self, x):
        if len(self.input_size) > 1:
            x = x.view(x.shape[0], -1)
        x = self.local_model_softmax_fc_1(x)
        x = self.relu(x)
        x = self.local_model_softmax_fc_2(x)
        return x