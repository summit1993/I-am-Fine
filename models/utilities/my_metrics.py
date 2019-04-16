# -*- coding: UTF-8 -*-

def Top_K_Right(true_labels, predictions):
    nums = len(true_labels)
    right = 0.0
    for i in range(nums):
        true_label = int(true_labels[i])
        if true_label in predictions[i]:
            right += 1.0
    # return right * 1.0  / nums
    return right
