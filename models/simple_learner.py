from torch import nn

from models.incremental_base import MultiTaskLearner
from models.resnet import get_resnet


class SimpleLearner(MultiTaskLearner):

    def __init__(self, loss, resnet_type="32", num_classes=10):
        super(SimpleLearner, self).__init__(loss, resnet_type=resnet_type, num_classes=num_classes)

        self.features_extractor = get_resnet(resnet_type)
        self.features_extractor.fc = nn.Sequential()

    def forward_and_compute_loss(self, images, labels):
        # Forward pass to the network
        outputs = self(images)

        loss = self.loss.compute_loss(   # It just use the classification loss
            outputs,
            labels,
        )
        return outputs, loss

    def before_task(self, train_loader, targets, val_loader=None, use_bias=False):
        super().before_task(train_loader, targets, val_loader, use_bias)


