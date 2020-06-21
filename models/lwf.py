import copy

import torch

import torch.nn as nn
from models.incremental_base import MultiTaskLearner
from models.resnet import get_resnet
from models.utils.utilities import classification_and_distillation_loss


class LwF(MultiTaskLearner):
    def __init__(self, resnet_type="32", num_classes=10):
        super(LwF, self).__init__(num_classes=num_classes)

        self.features_extractor = get_resnet(resnet_type)
        self.features_extractor.fc = nn.Sequential()

        self.previous_model = None

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

    def classify(self, x):
        _, preds = torch.max(torch.softmax(self.forward(x), dim=1), dim=1, keepdim=False)

        return preds

    def before_task(self, train_loader, targets, val_loader=None, use_bias=False):
        super().before_task(train_loader, targets, val_loader, use_bias)

        if self.n_known > 0:
            self.previous_model = copy.deepcopy(self.features_extractor)
            self.previous_model.fc = copy.deepcopy(self.classifier)
            self.previous_model.train(False)
            for param in self.previous_model.parameters():
                param.requires_grad = False

    def forward_and_compute_loss(self, images, labels):
        # Forward pass to the network
        outputs = self(images)

        previous_output = None
        if self.n_known > 0:
            assert previous_output is not None
            previous_output = self.previous_model(images)

        loss = classification_and_distillation_loss(
                outputs,
                labels,
                previous_output=previous_output,
                new_idx=self.n_known
        )
        return outputs, loss



