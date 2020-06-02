import copy

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config
from models.incremental_base import MultiTaskLearner
from models.resnet import get_resnet
from models.utils import l2_normalize
from models.utils.utilities import timer, remove_row, classification_and_distillation_loss, to_onehot


class iCaRL(MultiTaskLearner):
    """Implementation of iCarl.
    # References:
    - iCaRL: Incremental Classifier and Representation Learning
      Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
      https://arxiv.org/abs/1611.07725

      https://github.com/imyzx2017/icarl.pytorch/blob/master/icarl.py
    """

    def __init__(self, resnet_type="32", num_classes=10, k=2000):
        super(iCaRL, self).__init__()

        self.k = k
        self.n_classes = num_classes
        self.n_known = 0

        self.features_extractor = get_resnet(resnet_type)
        self.features_extractor.fc = nn.Sequential()
        self.classifier = nn.Linear(self.features_extractor.out_dim, num_classes)

        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.01)

        self.exemplars = {}
        self.exemplars_means = None

        self.previous_model = None

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

    def classify(self, batch_images):
        assert self.exemplars_means is not None
        assert self.exemplars_means.shape[0] == self.n_classes

        features = self.features_extractor(batch_images)
        features = l2_normalize(features)
        return self._nearest_prototype(self.exemplars_means, features)

    @staticmethod
    def _nearest_prototype(centers, features):
        pred_labels = []

        for feature in features:
            distances = torch.pow(centers - feature, 2).sum(-1)
            pred_labels.append(distances.argmin().item())

        return np.array(pred_labels)

    @staticmethod
    def _get_closest_feature(center, features):
        normalized_features = l2_normalize(features)
        distances = torch.pow(center - normalized_features, 2).sum(-1)
        return distances.argmin().item()

    @timer
    def build_exemplars(self, class_loader, class_index):
        exemplars = []

        self.eval()
        with torch.no_grad():
            features, class_mean = self._extract_features_and_mean(class_loader)

        self._expand_exemplars_means(class_index, class_mean)
        exemplars_feature_sum = torch.zeros((self.features_extractor.out_dim,)).to(Config.DEVICE)

        for k in range(min(self._m, len(features))):
            # argmin(class_mean - 1/k * (features + exemplars_sum))
            # TODO: checkear porque esta implementacion usa normalizacion_l2 en vez de dividir por k.
            #       Se entiende que al hacer (exemplars_feature_sum / k) daria como resultado
            #       exemplars_feature_mean. Porque el paper ademas hace features / k?
            idx = self._get_closest_feature(class_mean, features + exemplars_feature_sum)

            exemplars.append(class_loader.dataset.__getitem__(idx))
            exemplars_feature_sum += features[idx]

            # TODO: en el paper no quita los features ya agregados.
            features = remove_row(features, idx)

        self.exemplars[class_index] = exemplars

    def _extract_features_and_mean(self, dataloader):
        sum = torch.zeros((self.features_extractor.out_dim,)).to(Config.DEVICE)
        features = []
        qty = 0

        for images, _ in dataloader:
            images = images.to(Config.DEVICE)
            features.append(self.features_extractor(images))

            # Sum all
            sum += features[-1].sum(0)
            # Count how many
            qty += features[-1].shape[0]    # it is a batch

        mean = sum / qty
        features = torch.cat(features)

        return l2_normalize(features), l2_normalize(mean)

    def _expand_exemplars_means(self, class_idx, mean):
        if self.exemplars_means is None:
            assert class_idx == 0
            self.exemplars_means = mean[None, ...]
        else:
            # Checking if the new class follows the previous ones
            assert self.exemplars_means.shape[0] == class_idx, (self.exemplars_means.shape, class_idx)
            self.exemplars_means = torch.cat((self.exemplars_means, mean[None, ...]))

    @property
    def _m(self):
        """Returns the number of exemplars per class."""
        return self.k // self.n_classes

    @timer
    def reduce_exemplars(self):
        for class_idx in range(len(self.exemplars)):
            self.exemplars[class_idx] = self.exemplars[class_idx][:self._m]

    def _add_n_classes(self, n):
        """Add n classes in the final fc layer"""
        self.n_classes += n

        weight = self.classifier.weight.data
        bias = self.classifier.bias.data

        self.classifier = nn.Linear(self.features_extractor.out_dim, self.n_classes, bias=False)

        # TODO: Check initializations
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.01)

        self.classifier.weight.data[:self.n_classes - n] = weight
        self.classifier.bias.data[:self.n_classes - n] = bias

    def before_task(self, train_loader, val_loader=None):
        if self.n_known > 0:
            n = len(set(train_loader.dataset.targets))
            self._add_n_classes(n)

    def train_task(self, train_loader, optimizer, scheduler, num_epochs, val_loader=None, log_dir=None):
        self.to(Config.DEVICE)  # this will bring the network to GPU if DEVICE is cuda

        cudnn.benchmark  # Calling this optimizes runtime
        current_step = 0
        # Start iterating over the epochs
        for epoch in range(num_epochs):
            print('Starting epoch {}/{}, LR = {}'.format(epoch + 1, num_epochs, scheduler.get_last_lr()))

            # Iterate over the dataset
            for images, labels in train_loader:
                # Bring data over the device of choice
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)

                self.train()  # Sets module in training mode

                optimizer.zero_grad()  # Zero-ing the gradients

                labels_onehot = to_onehot(labels, self.n_classes).to(Config.DEVICE)

                # Forward pass to the network
                outputs = self(images)

                previous_output = None
                if self.previous_model is not None:
                    previous_output = self.previous_model(images)

                clf_loss, distil_loss = classification_and_distillation_loss(
                        outputs,
                        labels_onehot,
                        previous_output=previous_output,
                        new_idx=self.n_known
                    )
                loss = clf_loss + distil_loss
                loss.backward()  # backward pass: computes gradients
                optimizer.step()  # update weights based on accumulated gradients

                current_step += 1

            # Step the scheduler
            scheduler.step()

    def after_task(self, train_loader):
        self.reduce_exemplars()
        for class_idx in set(train_loader.dataset.targets):
            idx = train_loader.dataset.get_class_indices(class_idx)
            class_data = Subset(train_loader.dataset, np.where(idx == 1)[0])
            class_loader = DataLoader(class_data, batch_size=8, shuffle=True)
            self.build_exemplars(class_loader, class_idx)

        self.n_known = self.n_classes
        self.previous_model = copy.deepcopy(self.features_extractor)
        self.previous_model.fc = copy.deepcopy(self.classifier)
        self.previous_model.train(False)

    def eval_task(self, eval_loader):
        self.to(Config.DEVICE)  # this will bring the network to GPU if DEVICE is cuda
        self.train(False)  # Set Network to evaluation mode

        running_corrects = 0
        for images, labels in tqdm(eval_loader):
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            # Get predictions
            preds = self.classify(images)

            running_corrects += torch.sum(preds == labels.data).data.item()

        # Calculate Accuracy
        accuracy = running_corrects / float(len(eval_loader.dataset))
        return accuracy
