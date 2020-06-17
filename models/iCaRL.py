import copy

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from datasets.cifar import get_class_dataset
from datasets.common_datasets import SimpleDataset
from models.incremental_base import MultiTaskLearner
from models.resnet import get_resnet
from models.utils import l2_normalize
from models.utils.utilities import timer, remove_row, classification_and_distillation_loss, to_onehot, \
    class_dist_loss_icarl, ReverseIdxSorted


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

        self.exemplars = {}
        self.exemplars_means = None

        self.previous_model = None

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x
    """
    def classify(self, batch_images):
        assert self.exemplars_means is not None
        assert self.exemplars_means.shape[0] == self.n_classes

        self.features_extractor.train(False)

        features = self.features_extractor(batch_images)
        return self._nearest_prototype(self.exemplars_means, features)

    @staticmethod
    def _nearest_prototype(centers, features):
        batch_size = features.size(0)
        centers = torch.stack([centers] * batch_size)   # (batch_size, n_classes, feature_size)
        centers = centers.transpose(1, 2)   # (batch_size, feature_size, n_classes)

        normalized_features = []
        for feature in features:
            normalized_features.append(l2_normalize(feature))
        normalized_features = torch.stack(normalized_features, dim=0)
        normalized_features = normalized_features.unsqueeze(2)  # (batch_size, feature_size, 1)
        normalized_features = normalized_features.expand_as(centers)    # (batch_size, feature_size, n_classes)

        preds = torch.argmin((normalized_features - centers).pow(2).sum(1), dim=1)
        return preds

    """
    def classify(self, batch_images):
        assert self.exemplars_means is not None
        assert self.exemplars_means.shape[0] == self.n_classes

        self.features_extractor.train(False)

        features = self.features_extractor(batch_images)

        normalized_features = []
        for feature in features:
            normalized_features.append(l2_normalize(feature))
        normalized_features = torch.stack(normalized_features, dim=0)

        return self._nearest_prototype(self.exemplars_means, normalized_features)

    @staticmethod
    def _nearest_prototype(centers, features):
        pred_labels = []

        for feature in features:
            distances = torch.pow(centers - feature, 2).sum(-1)
            pred_labels.append(distances.argmin().item())

        return torch.from_numpy(np.array(pred_labels))

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

        reverse_index = ReverseIdxSorted(len(features))

        for k in range(min(self._m, len(features))):
            # argmin(class_mean - 1/k * (features + exemplars_sum))
            # TODO: checkear porque esta implementacion usa normalizacion_l2 en vez de dividir por k.
            #       Se entiende que al hacer (exemplars_feature_sum / k) daria como resultado
            #       exemplars_feature_mean. Porque el paper ademas hace features / k?
            idx = self._get_closest_feature(class_mean, features + exemplars_feature_sum)
            true_idx = reverse_index[idx]

            exemplars.append(class_loader.dataset[true_idx][0])
            exemplars_feature_sum += features[idx]

            # TODO: en el paper no quita los features ya agregados.
            features = remove_row(features, idx)

        self.exemplars[class_index] = exemplars

    """
    def _extract_features_and_mean(self, dataloader):
        features = []

        for images, _ in dataloader:
            images = images.to(Config.DEVICE)
            feature = self.features_extractor(images)

            feature = feature / np.linalg.norm(feature.cpu())  # Normalize

            features.append(feature)

        features = torch.cat(features)
        mean = features.mean(0)
        mean = mean / np.linalg.norm(mean.cpu())  # Normalize

        return features, mean
    """

    def _extract_features_and_mean(self, dataloader):
        sum = torch.zeros((self.features_extractor.out_dim,)).to(Config.DEVICE)
        features = []
        qty = 0

        for images, _ in dataloader:
            images = images.to(Config.DEVICE)
            features.append(l2_normalize(self.features_extractor(images)))

            # Sum all
            sum += features[-1].sum(0)
            # Count how many
            qty += features[-1].shape[0]    # it is a batch

        mean = sum / qty
        features = torch.cat(features)

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

    def _add_n_classes(self, n):
        """Add n classes in the final fc layer"""
        self.n_classes += n

        weight = self.classifier.weight.data
        bias = self.classifier.bias.data

        self.classifier = nn.Linear(self.features_extractor.out_dim, self.n_classes)

        self.classifier.weight.data[:self.n_classes - n] = weight
        self.classifier.bias.data[:self.n_classes - n] = bias

    def combine_training_exemplars(self, train_loader):
        datasets = [train_loader.dataset]
        for class_idx in range(len(self.exemplars)):
            datasets.append(SimpleDataset(self.exemplars[class_idx], [class_idx] * len(self.exemplars[class_idx])))

        new_train_loader = DataLoader(ConcatDataset(datasets), batch_size=train_loader.batch_size,
                                      shuffle=True, num_workers=Config.NUM_WORKERS)
        return new_train_loader

    def before_task(self, train_loader, targets, val_loader=None):
        if self.n_known > 0:
            n = len(set(targets))
            self._add_n_classes(n)

            self.previous_model = copy.deepcopy(self.features_extractor)
            self.previous_model.fc = copy.deepcopy(self.classifier)
            self.previous_model.train(False)
            for param in self.previous_model.parameters():
                param.requires_grad = False

            print('adding {} classes, total {}'.format(n, self.n_classes))

    def train_task(self, train_loader, optimizer, scheduler, num_epochs, val_loader=None, log_dir=None):
        self.to(Config.DEVICE)  # this will bring the network to GPU if DEVICE is cuda

        if log_dir is not None:
            # TensorboardX summary writer
            tb_writer = SummaryWriter(log_dir=log_dir)

        train_exemplars_loader = self.combine_training_exemplars(train_loader)

        cudnn.benchmark  # Calling this optimizes runtime
        current_step = 0
        # Start iterating over the epochs
        for epoch in range(num_epochs):
            print('Starting epoch {}/{}, LR = {}'.format(epoch + 1, num_epochs, scheduler.get_last_lr()))

            # Iterate over the dataset
            for images, labels in train_exemplars_loader:
                # Bring data over the device of choice
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)

                self.train()  # Sets module in training mode

                optimizer.zero_grad()  # Zero-ing the gradients

                # Forward pass to the network
                outputs = self(images)

                previous_output = None
                if self.previous_model is not None:
                    previous_output = self.previous_model(images)

                loss = class_dist_loss_icarl(
                        outputs,
                        labels,
                        previous_output=previous_output,
                        new_idx=self.n_known
                    )

                # Log the information and add to tensorboard
                if current_step % Config.LOG_FREQUENCY == 0:
                    with torch.no_grad():
                        _, preds = torch.max(outputs, 1)
                        accuracy = torch.sum(preds == labels) / float(len(labels))

                        print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {:.4f}'
                              .format(epoch + 1, current_step, loss.item(), accuracy.item()))

                        if log_dir is not None:
                            tb_writer.add_scalar('loss', loss.item(), current_step)
                            tb_writer.add_scalar('accuracy', accuracy.item(), current_step)
                            #tb_writer.add_scalar('class_loss', clf_loss.item(), current_step)
                            #tb_writer.add_scalar('dis_loss', distil_loss.item(), current_step)

                loss.backward()  # backward pass: computes gradients
                optimizer.step()  # update weights based on accumulated gradients

                current_step += 1

            # Step the scheduler
            scheduler.step()

    def after_task(self, train_loader, targets):
        self.reduce_exemplars()
        for class_idx in sorted(set(targets)):
            class_data = get_class_dataset(train_loader.dataset, class_idx)
            class_loader = DataLoader(class_data, batch_size=Config.BATCH_SIZE, shuffle=False)
            self.build_exemplars(class_loader, class_idx)

        self.n_known = self.n_classes

    def eval_task(self, eval_loader):
        self.to(Config.DEVICE)  # this will bring the network to GPU if DEVICE is cuda
        self.train(False)  # Set Network to evaluation mode

        running_corrects = 0
        for images, labels in tqdm(eval_loader):
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            # Get predictions
            preds = self.classify(images).to(Config.DEVICE)

            running_corrects += torch.sum(preds == labels.data).data.item()

        # Calculate Accuracy
        accuracy = running_corrects / float(len(eval_loader.dataset))
        return accuracy

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
