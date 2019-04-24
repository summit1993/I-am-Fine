# -*- coding: UTF-8 -*-
from HC.local_model.local_softmax_model import *

def get_local_model(name, **kwargs):
    if name == 'softmax':
        return LocalSoftmaxModel(**kwargs)
    return None