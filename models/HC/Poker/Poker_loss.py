# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

def Poker_loss(outputs, true_labels, device, hierarchy, loss_fn='softmax'):
    if loss_fn == 'softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    samples_count = len(true_labels)
    total_loss = 0.0
    nodes = hierarchy['nodes']
    paths = hierarchy['paths']
    for code in outputs:
        output = outputs[code]
        node = nodes[code]
        children_list = node.get_children_code()
        children_num = len(children_list)
        children_set = set(children_list)
        if loss_fn == 'softmax':
            node_labels = torch.zeros(samples_count, dtype=torch.long)
        else:
            node_labels = torch.zeros((samples_count, len(children_list)))
        for i in range(samples_count):
            true_label = true_labels[i].item()
            u_set = paths[true_label] & children_set
            if len(u_set) > 0:
                child_index = children_list.index(list(u_set)[0])
                if loss_fn == 'softmax':
                    node_labels[i] = child_index
                else:
                    node_labels[i][child_index] = 1.0
            else:
                if loss_fn == 'softmax':
                    node_labels[i] = children_num

        node_labels = node_labels.to(device)
        loss = criterion(output, node_labels)
        total_loss += loss

    return total_loss * 1.0 / len(outputs)