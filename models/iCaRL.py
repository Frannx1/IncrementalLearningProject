import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from config import Config
from models.incremental_base import MultiTaskLearner
from models.resnet import get_resnet
from models.utils import l2_normalize
from models.utils.utilities import timer, get_closest_feature, remove_row


class iCarl(MultiTaskLearner):
    """Implementation of iCarl.
    # References:
    - iCaRL: Incremental Classifier and Representation Learning
      Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
      https://arxiv.org/abs/1611.07725

      https://github.com/imyzx2017/icarl.pytorch/blob/master/icarl.py
    """

    def __init__(self, resnet_type="32", n_classes=10, k=2000):
        super(iCarl, self).__init__()

        self.k = k
        self.n_classes = n_classes

        self.features_extractor = get_resnet(resnet_type)
        self.classifier = nn.Linear(self.features_extractor.out_dim, n_classes)

        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.01)

        self.exemplars = {}
        self.exemplars_means = None

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

    def classify(self, batch_images):
        assert self.means is not None
        assert self.means.shape[0] == self.n_classes

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

    @timer
    def build_exemplars(self, class_loader, class_index):
        exemplars = []

        self.eval()
        with torch.no_grad():
            features, class_mean = self._extract_features_and_mean(class_loader)

        self._expand_exemplar_means(class_index, class_mean)
        exemplars_feature_sum = torch.zeros((self.features_extractor.out_dim,))

        for k in range(min(self._m, len(features))):
            # argmin(class_mean - 1/k * (features + exemplars_sum))
            # TODO: checkear porque esta implementacion usa normalizacion_l2 en vez de dividir por k.
            #       Se entiende que al hacer (exemplars_feature_sum / k) daria como resultado
            #       exemplars_feature_mean. Porque el paper ademas hace features / k?
            idx = get_closest_feature(class_mean, features + exemplars_feature_sum)

            exemplars.append(class_loader.dataset.__getitem__(idx))
            exemplars_feature_sum += features[idx]

            # TODO: en el paper no quita los features ya agregados.
            features = remove_row(features, idx)

        self.exemplars[class_index] = exemplars

    def _extract_features_and_mean(self, dataloader):
        sum = torch.zeros((self.features_extractor.out_dim,))
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

    def _expand_exemplar_means(self, class_idx, mean):
        if self.exemplar_means is None:
            assert class_idx == 0
            self.exemplar_means = mean[None, ...]
        else:
            # Checking if the new class follows the previous ones
            assert self.exemplar_means.shape[0] == class_idx, (self.exemplar_means.shape, class_idx)
            self.exemplar_means = torch.cat((self.exemplar_means, mean[None, ...]))

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
        n = set(train_loader.dataset.targets)
        self._add_n_classes(n)

    def train_task(self, train_loader, val_loader=None):

        for epoch in range(self._n_epochs):
            _loss, val_loss = 0., 0.

            self._scheduler.step()

            prog_bar = tqdm(train_loader)
            for i, (inputs, targets) in enumerate(prog_bar, start=1):
                self._optimizer.zero_grad()

                loss_graph = 0
                loss_2 = 0
                feature_metric = None

                if self._data_transform_memory is not None:
                    features = self._network.extract(
                        self._data_transform_memory.to(self._network.device)
                    )

                    feature_metric = torch.cdist(features, features)

                    print('feature_metric:', feature_metric)
                    print('feature_metric.shape:', feature_metric.shape)

                    # get the upper triangle matrix to calc the loss
                    first_part = self.metric_2.mul(torch.pow(feature_metric, 2))
                    second_part = (1 - self.metric_2).mul(
                        torch.pow(torch.clamp(self.margin - feature_metric, min=0.0), 2))
                    print('first_part:', first_part)
                    print('second_part:', second_part)
                    loss_graph += first_part + second_part
                    print('loss_graph:', loss_graph)
                    print('loss_graph.shape:', loss_graph.shape)
                    count = 0
                    for i in range(loss_graph.shape[0]):
                        for j in range(loss_graph.shape[1]):
                            if i >= j:
                                loss_2 += loss_graph[i][j]
                                count += 1
                    loss_2 = (loss_2 / count) / 2
                    print('loss_2:', loss_2)
                    print('loss_2.shape:', loss_2.shape)
                    loss_2.backward(retain_graph=True)

                # loss for classification and distillation
                loss = self._forward_loss(inputs, targets)

                if not utils._check_loss(loss):
                    import pdb
                    pdb.set_trace()

                print('loss:', loss)
                print('loss.shape:', loss.shape)
                loss.backward()

                self._optimizer.step()

                _loss += loss.item()

                if val_loader is not None and i == len(train_loader):
                    for inputs, targets in val_loader:
                        val_loss += self._forward_loss(inputs, targets).item()

                prog_bar.set_description(
                    "Task {}/{}, Epoch {}/{} => Clf loss: {}, Val loss: {}".format(
                        self._task + 1, self._n_tasks,
                        epoch + 1, self._n_epochs,
                        round(_loss / i, 3),
                        round(val_loss, 3)
                    )
                )

    def after_task(self, inc_dataset):
        self.build_examplars(inc_dataset)

        self._old_model = self._network.copy().freeze()

    def eval_task(self, data_loader):
        ypred, ytrue = compute_accuracy(self._network, data_loader, self._class_means)

        return ypred, ytrue
