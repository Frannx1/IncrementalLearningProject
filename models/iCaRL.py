import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from config import Config
from datasets.cifar import get_class_dataset
from datasets.common_datasets import SimpleDataset
from models.incremental_base import MultiTaskLearner
from models.resnet import get_resnet
from models.utils import l2_normalize
from models.utils.utilities import timer, remove_row, class_dist_loss_icarl, ReverseIdxSorted, \
    replace_row


class iCaRL(MultiTaskLearner):
    """Implementation of iCarl.
    # References:
    - iCaRL: Incremental Classifier and Representation Learning
      Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
      https://arxiv.org/abs/1611.07725

      https://github.com/imyzx2017/icarl.pytorch/blob/master/icarl.py
    """

    def __init__(self, resnet_type="32", num_classes=10, k=2000):
        super(iCaRL, self).__init__(num_classes=num_classes)

        self.features_extractor = get_resnet(resnet_type)
        self.features_extractor.fc = nn.Sequential()

        self.k = k
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

        self.features_extractor.train(False)

        features = self.features_extractor(batch_images)
        return self._nearest_prototype(self.exemplars_means, features)

    @staticmethod
    def _nearest_prototype(centers, features):
        pred_labels = []

        for feature in features:
            norm_feature = l2_normalize(feature)
            distances = torch.pow(centers - norm_feature, 2).sum(-1)
            pred_labels.append(distances.argmin().item())

        return torch.from_numpy(np.array(pred_labels))

    @staticmethod
    def _get_closest_feature(center, features):
        normalized_features = []
        for feature in features:
            normalized_features.append(l2_normalize(feature))

        normalized_features = torch.stack(normalized_features)
        distances = torch.pow(center - normalized_features, 2).sum(-1)
        return distances.argmin().item()

    @timer
    def build_exemplars(self, class_loader, class_index):
        exemplars = []

        self.eval()
        features, class_mean = self._extract_features_and_mean(class_loader)

        self._expand_exemplars_means(class_index, class_mean)

        exemplars_feature_sum = torch.zeros((self.features_extractor.out_dim,)).to(Config.DEVICE)
        reverse_index = ReverseIdxSorted(len(features))

        for k in range(min(self._m, len(features))):
            # argmin(class_mean - 1/k * (features + exemplars_sum))
            idx = self._get_closest_feature(class_mean, (1.0/(k+1)) * (features + exemplars_feature_sum))
            true_idx = reverse_index[idx]

            exemplars.append(class_loader.dataset[true_idx][0])
            exemplars_feature_sum += features[idx]
            features = remove_row(features, idx)

        self.exemplars[class_index] = exemplars

    def _extract_features_and_mean(self, dataloader):
        features = []

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(Config.DEVICE)
                features.append(l2_normalize(self.features_extractor(images)))

            features = torch.cat(features)
            mean = features.mean(dim=0)

        return features, l2_normalize(mean)

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

    @timer
    def recompute_exemplars_means(self):
        for class_idx in range(len(self.exemplars)):
            class_dataset = SimpleDataset(self.exemplars[class_idx], [class_idx] * len(self.exemplars[class_idx]))
            class_loader = DataLoader(class_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                                      num_workers=Config.NUM_WORKERS)
            _, class_mean = self._extract_features_and_mean(class_loader)
            self.exemplars_means = replace_row(self.exemplars_means, class_mean, class_idx)

    def augment_train_dataset(self, train_loader):
        datasets = [train_loader.dataset]
        for class_idx in range(len(self.exemplars)):
            datasets.append(SimpleDataset(self.exemplars[class_idx], [class_idx] * len(self.exemplars[class_idx])))

        new_train_loader = DataLoader(ConcatDataset(datasets), batch_size=train_loader.batch_size,
                                      shuffle=True, num_workers=Config.NUM_WORKERS)
        return new_train_loader

    def before_task(self, train_loader, targets, val_loader=None, use_bias=True):
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

        loss = class_dist_loss_icarl(
                outputs,
                labels,
                previous_output=previous_output,
                new_idx=self.n_known
        )
        return outputs, loss

    def after_task(self, train_loader, targets):
        self.to(Config.DEVICE)
        self.train(False)

        self.recompute_exemplars_means()
        self.reduce_exemplars()
        for class_idx in sorted(set(targets)):
            class_dataset = get_class_dataset(train_loader.dataset, class_idx)
            class_loader = DataLoader(class_dataset, batch_size=train_loader.batch_size, shuffle=False)
            self.build_exemplars(class_loader, class_idx)

        super().after_task(train_loader, targets)

    def eval_hybrid1(self, eval_loader):
        self.to(Config.DEVICE)  # this will bring the network to GPU if DEVICE is cuda
        self.train(False)  # Set Network to evaluation mode

        running_corrects = 0
        for images, labels in tqdm(eval_loader):
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            class_pred = self(images)

            # Get predictions
            _, preds = torch.max(class_pred.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

        # Calculate Accuracy
        accuracy = running_corrects / float(len(eval_loader.dataset))
        return accuracy
