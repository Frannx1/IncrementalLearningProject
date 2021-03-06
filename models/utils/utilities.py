import time

import torch
from torch import nn
from torch.nn import functional as F

from config import Config


def l2_normalize(tensor, dim=None):
    return F.normalize(tensor, p=2, dim=dim)


def remove_row(matrix, row_idx):
    return torch.cat((
        matrix[:row_idx, ...],
        matrix[row_idx + 1:, ...]
    ))


def replace_row(matrix, tensor, row_idx):
    return torch.cat((
        matrix[:row_idx, ...],
        tensor[None, ...],
        matrix[row_idx + 1:, ...]
    ))


def to_onehot(labels, n_classes):
    return torch.eye(n_classes)[labels]


def timer(func):
    def _timer(*args, **kwargs):
        print("Doing <{}>...".format(func.__name__), end=" ")
        tic = time.time()
        res = func(*args, **kwargs)
        print("Done in {:.2f}".format(time.time() - tic))

        return res
    return _timer


def distillation_loss(logits, old_logits):
    # The distillation loss is computed over all the previous already known classes
    assert len(logits) == len(old_logits)

    return sum(
        q * torch.log(g) + (1 - q) * torch.log(1 - g)
        for q, g in zip(F.sigmoid(old_logits), F.sigmoid(logits))
    )


def classification_and_distillation_loss(outputs, labels, previous_output=None, new_idx=0):
    outputs, labels = outputs.to(Config.DEVICE), labels.to(Config.DEVICE)
    labels_onehot = to_onehot(labels, outputs.shape[1]).to(Config.DEVICE)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    clf_loss = criterion(outputs[:, new_idx:], labels_onehot[:, new_idx:])

    if new_idx > 0:
        assert previous_output is not None
        previous_output = previous_output.to(Config.DEVICE)
        distil_loss = criterion(outputs[:, :new_idx], torch.sigmoid(previous_output[:, :new_idx]))
    else:
        # First learning no distillation loss
        distil_loss = torch.zeros(1, requires_grad=False).to(Config.DEVICE)

    return clf_loss + distil_loss


def class_dist_loss_icarl(outputs, labels, previous_output=None, new_idx=0):
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    labels_onehot = to_onehot(labels, outputs.shape[1]).to(Config.DEVICE)

    if new_idx > 0:
        assert previous_output is not None
        target = torch.cat((torch.sigmoid(previous_output[:, :new_idx]), labels_onehot[:, new_idx:]), dim=1)
    else:
        target = labels_onehot

    return criterion(outputs, target)


class ReverseIdxSorted:

    def __init__(self, length):
        self.item_list = [x for x in range(length)]

    def __getitem__(self, index):
        # Returns the true index of a list in
        # which some of its elements were removed.
        # Every time an item is got, it
        # is considered as removed

        item = self.item_list[index]
        self.item_list.remove(item)

        return item

