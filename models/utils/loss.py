from abc import abstractmethod, ABC

import torch
from torch import nn

from models.utils.utilities import to_onehot


class CrossEntropyLossOneHot(nn.CrossEntropyLoss):

    def forward(self, input, target):
        target = torch.argmax(target, dim=1)
        return super().forward(input, target)


class MaximizeCosineSimilarityLoss(nn.CosineEmbeddingLoss):

    def forward(self, input1, input2, target=None):
        target = torch.ones(input1.shape).to(input1.device)
        return super().forward(input1, input2, target)


class Loss(ABC):

    @abstractmethod
    def compute_loss(self, outputs, labels, previous_output, new_idx, features, old_features):
        pass


class ClassDistLoss(Loss):

    def __init__(self, class_loss, dist_loss, balanced, dist_feature, device):
        self.class_loss = class_loss
        self.dist_loss = dist_loss
        self.balanced = balanced
        self.dist_feature = dist_feature
        self.device = device

    def compute_loss(self, outputs, labels, previous_output=None, new_idx=0, features=None, old_features=None):
        class_factor, dist_factor = 1.0, 1.0

        labels_onehot = to_onehot(labels, outputs.shape[1]).to(self.device)
        class_loss = self.class_loss(outputs[:, new_idx:], labels_onehot[:, new_idx:])

        if new_idx > 0:
            if self.dist_feature:
                assert features is not None
                assert old_features is not None
                dist_loss = self.dist_loss(features, old_features)
            else:
                assert previous_output is not None
                dist_loss = self.dist_loss(outputs[:, :new_idx], torch.sigmoid(previous_output[:, :new_idx]))

            if self.balanced:
                class_factor = (outputs.shape[1] - new_idx) / float(outputs.shape[1])
                dist_factor = new_idx / float(outputs.shape[1])
        else:
            # First learning no distillation loss
            dist_loss = torch.zeros(1, requires_grad=False).to(self.device)

        return class_factor * class_loss + dist_factor * dist_loss


class ClassDistLossBuilder:

    @staticmethod
    def build(device, class_loss='bce', dist_loss='bce', balanced=True):
        class_loss, _ = ClassDistLossBuilder.get_loss_conf(class_loss, 'class')
        dist_loss, dist_feature = ClassDistLossBuilder.get_loss_conf(dist_loss, 'dist')
        double_loss = ClassDistLoss(class_loss, dist_loss, balanced, dist_feature, device)
        return double_loss

    @staticmethod
    def get_loss_conf(loss, type):
        # Returns the loss and if it needs the features output or not (the fc output).
        loss = loss.lower()
        if loss == 'l2':
            if type == 'dist':
                return nn.MSELoss(), True
            else:
                raise ValueError('It is not possible to have a L2 loss for classification.')
        if loss == 'ce':
            return CrossEntropyLossOneHot(), False
        if loss == 'bce':
            return nn.BCEWithLogitsLoss(), False
        if loss == 'cos':
            if type == 'dist':
                return MaximizeCosineSimilarityLoss(), True
            else:
                raise ValueError('It is not possible to have a Cosine Embedding loss for classification.')
        raise NotImplementedError()


