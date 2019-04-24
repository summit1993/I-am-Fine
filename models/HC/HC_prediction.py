# -*- coding: UTF-8 -*-
import queue
import torch
import torch.nn.functional as F

def HC_prediction(outputs, hierarchy, fn='sotmax'):
    que = queue.Queue()
    leaf_index_map = hierarchy['leaf_index_map']
    path_length = {}
    nodes = hierarchy['nodes']
    code_score_dict = {}
    predictions = torch.zeros(outputs[-1].shape[0], len(leaf_index_map))
    que.put(-1)
    path_length[-1] = 0
    while not que.empty():
        code = que.get()
        node = nodes[code]
        children_code_list = node.get_children_code()
        children_count = len(children_code_list)
        p_len = path_length[code]
        if children_count > 0:
            if children_count > 1:
                if fn == 'softmax':
                    output = F.log_softmax(outputs[code], dim=1)
                elif fn == 'BCE':
                    output = torch.log(torch.sigmoid(outputs[code]))
                children_scores = output.transpose(1, 0) + code_score_dict[code]
                children_scores = children_scores.transpose(1, 0)
                for i in range(children_count):
                    child = children_code_list[i]
                    code_score_dict[child] = children_scores[:, i]
                    path_length[child] = p_len + 1
                    que.put(child)
            else:
                child = children_code_list[0]
                code_score_dict[child] = code_score_dict[code]
                path_length[child] = p_len
                que.put(child)
        else:
            predictions[:, leaf_index_map[code]] = code_score_dict[code] / p_len
    return predictions
