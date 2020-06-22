from abc import abstractmethod, ABC

import torch
from torch import nn

from models.utils.utilities import to_onehot


class CrossEntropyLossOneHot(nn.CrossEntropyLoss):

    def forward(self, input, target):
        target = torch.argmax(target, dim=1)
        return super().forward(input, target)


class Loss(ABC):

    @abstractmethod
    def compute_loss(self, outputs, labels, previous_output, new_idx):
        pass


class DoubleLoss(Loss):

    def __init__(self, class_loss, dist_loss, balanced, device):
        self.class_loss = class_loss
        self.dist_loss = dist_loss
        self.balanced = balanced
        self.device = device

    def compute_loss(self, outputs, labels, previous_output=None, new_idx=0):
        class_factor, dist_factor = 1.0, 1.0

        labels_onehot = to_onehot(labels, outputs.shape[1]).to(self.device)
        class_loss = self.class_loss(outputs[:, new_idx:], labels_onehot[:, new_idx:])

        if new_idx > 0:
            assert previous_output is not None
            dist_loss = self.dist_loss(outputs[:, :new_idx], torch.sigmoid(previous_output[:, :new_idx]))

            if self.balanced:
                class_factor = (outputs.shape[1] - new_idx) / float(outputs.shape[1])
                dist_factor = new_idx / float(outputs.shape[1])
        else:
            # First learning no distillation loss
            dist_loss = torch.zeros(1, requires_grad=False).to(self.device)

        return class_factor * class_loss + dist_factor * dist_loss


class DoubleLossBuilder:

    @staticmethod
    def build(device, class_loss='bce', dist_loss='bce', balanced=True):
        class_loss = DoubleLossBuilder._get_loss(class_loss)
        dist_loss = DoubleLossBuilder._get_loss(dist_loss)
        double_loss = DoubleLoss(class_loss, dist_loss, balanced, device)
        return double_loss

    @staticmethod
    def _get_loss(loss):
        loss = loss.lower()
        if loss == 'l2':
            return nn.MSELoss()
        if loss == 'ce':
            return CrossEntropyLossOneHot()
        if loss == 'bce':
            return nn.BCEWithLogitsLoss()

        raise NotImplementedError()

