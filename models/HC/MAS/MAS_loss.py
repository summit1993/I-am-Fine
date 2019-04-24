# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn

def MAS_loss(outputs, true_labels, device, hierarchy, use_all=True):
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    samples_count = len(true_labels)
    total_loss = 0.0
    nodes = hierarchy['nodes']
    paths = hierarchy['paths']
    for code in outputs:
        output = outputs[code]
        node = nodes[code]
        children_list = node.get_children_code()
        children_set = set(children_list)
        node_labels = torch.zeros((samples_count, len(children_list)))
        if not use_all:
            weight_instance = torch.ones(samples_count)
            positive_count = 0.0
        for i in range(samples_count):
            true_label = true_labels[i].item()
            u_set = paths[true_label] & children_set
            if len(u_set) > 0:
                child_index = children_list.index(list(u_set)[0])
                node_labels[i][child_index] = 1.0
                if not use_all:
                    positive_count += 1.0
            else:
                if not use_all:
                    weight_instance[i] = 0.0

        node_labels = node_labels.to(device)
        loss = criterion(output, node_labels)
        if not use_all:
            weight_instance = weight_instance.to(device)
            loss = loss.transpose(1, 0) * weight_instance
            loss = loss.transpose(1, 0)
            loss = loss.sum()
            if positive_count > 0:
                loss = loss / positive_count
        else:
            loss = loss.sum()
            loss = loss / samples_count

        total_loss += loss

    return total_loss * 1.0 / len(outputs)
