from abc import abstractmethod

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from models.resnet import get_resnet


class MultiTaskLearner(nn.Module):
    """Base incremental learner.
    Methods are called in this order (& repeated for each new task):
    1. before_task
    2. train_task
    3. after_task
    4. eval_task

    Reference: https://github.com/AfricanxAdmiral/icarl/blob/master/inclearn/models/base.py
    """
    def __init__(self, loss, resnet_type="32", num_classes=10):
        super(MultiTaskLearner, self).__init__()
        self.n_classes = num_classes
        self.n_known = 0

        self.loss = loss

        self.features_extractor = get_resnet(resnet_type)
        self.features_extractor.fc = nn.Sequential()
        self.classifier = nn.Linear(self.features_extractor.out_dim, num_classes)

    def forward(self, x):
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

    def classify(self, batch_images):
        class_pred = self(batch_images)
        _, preds = torch.max(class_pred.data, 1)
        return preds

    def before_task(self, train_loader, targets, val_loader, use_bias):
        if self.n_known > 0:
            n = len(set(targets))
            self._add_n_classes(n, use_bias)
            print('Adding {} classes, total {}'.format(n, self.n_classes))

    def train_task(self, train_loader, optimizer, scheduler, num_epochs, val_loader=None, log_dir=None):
        self.to(Config.DEVICE)  # this will bring the network to GPU if DEVICE is cuda

        if log_dir is not None:
            # TensorboardX summary writer
            tb_writer = SummaryWriter(log_dir=log_dir)

        train_exemplars_loader = self.augment_train_dataset(train_loader)

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

                outputs, loss = self.forward_and_compute_loss(images, labels)

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

                loss.backward()  # backward pass: computes gradients
                optimizer.step()  # update weights based on accumulated gradients

                current_step += 1

            # Step the scheduler
            scheduler.step()

        if log_dir is not None:
            tb_writer.close()

    def after_task(self, train_loader, targets):
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

    def _add_n_classes(self, n, use_bias):
        """Add n classes in the final fc layer"""
        self.n_classes += n

        weight = self.classifier.weight.data
        if use_bias:
            bias = self.classifier.bias.data

        self.classifier = nn.Linear(self.features_extractor.out_dim, self.n_classes, use_bias)

        self.classifier.weight.data[:self.n_classes - n] = weight
        if use_bias:
            self.classifier.bias.data[:self.n_classes - n] = bias

    def augment_train_dataset(self, train_loader):
        return train_loader

    @abstractmethod
    def forward_and_compute_loss(self, images, labels):
        pass
