import time

import numpy as np
import torch


def l2_normalize(tensor):
    return tensor / torch.norm(tensor, p=2)


def remove_row(matrix, row_idx):
    return torch.cat((
        matrix[:row_idx, ...],
        matrix[row_idx + 1:, ...]
    ))


def get_closest_feature(center, features):
    normalized_features = l2_normalize(features)
    distances = torch.pow(center - normalized_features, 2).sum(-1)
    return distances.argmin().item()


def timer(func):
    def _timer(*args, **kwargs):
        print("Doing <{}>...".format(func.__name__), end=" ")
        tic = time.time()
        res = func(*args, **kwargs)
        print("Done in {:.2f}".format(time.time() - tic))

        return res
    return _timer
