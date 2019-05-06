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
    loss_num  = 0
    for code in outputs:
        output = outputs[code]
        node = nodes[code]
        children_list = node.get_children_code()
        children_set = set(children_list)
        node_labels = []
        node_labels_index = []
        for i in range(samples_count):
            true_label = true_labels[i].item()
            u_set = paths[true_label] & children_set
            if len(u_set) > 0:
                child_index = children_list.index(list(u_set)[0])
                node_labels.append(child_index)
                node_labels_index.append(i)

        node_labels_num = len(node_labels)
        if node_labels_num == 0:
            continue
        output = output[node_labels_index]

        if loss_fn == 'softmax':
            true_labels = torch.zeros(node_labels_num, dtype=torch.long)
        else:
            true_labels = torch.zeros((node_labels_num, len(children_list)))

        for i in range(node_labels_num):
            label = node_labels[i]
            if loss_fn == 'softmax':
                true_labels[i] = label
            else:
                true_labels[i][label] = 1.0

        true_labels = true_labels.to(device)
        loss = criterion(output, true_labels)
        loss_num += 1
        total_loss += loss

    if loss_num > 0:
        return total_loss * 1.0 / loss_num
    else:
        return total_loss