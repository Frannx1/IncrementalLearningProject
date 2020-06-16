import os

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from config import Config

from torch.nn import functional as F

from models.utils.utilities import to_onehot


def lwf_train_model(net, train_dataloader, criterion, optimizer, scheduler, num_epochs, n_known_classes, log_dir=None,
                    previous_model=None):
    # By default, everything is loaded to cpu
    net = net.to(Config.DEVICE)  # this will bring the network to GPU if DEVICE is cuda

    if n_known_classes > 0:
        assert previous_model is not None
        previous_model = previous_model.to(Config.DEVICE)
        previous_model.train(False)

    cudnn.benchmark  # Calling this optimizes runtime

    if log_dir is not None:
        # TensorboardX summary writer
        params_save = 'lr: {}, batch: {}, epochs: {}'.format(scheduler.get_last_lr()[0],
                                                             train_dataloader.batch_size,
                                                             num_epochs)

        log_dir = os.path.join(log_dir, params_save)
        tb_writer = SummaryWriter(log_dir=log_dir)

    current_step = 0
    # Start iterating over the epochs
    for epoch in range(num_epochs):
        print('Starting epoch {}/{}, LR = {}'.format(epoch + 1, num_epochs, scheduler.get_last_lr()))

        # Iterate over the dataset
        for images, labels in train_dataloader:
            # Bring data over the device of choice
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)

            net.train()  # Sets module in training mode

            # Zero-ing the gradients
            optimizer.zero_grad()

            # Forward pass to the network
            outputs = net(images)
            labels_onehot = to_onehot(labels, outputs.shape[1]).to(Config.DEVICE)
            criterion_class = nn.BCEWithLogitsLoss(reduction='mean')
            loss = criterion_class(outputs[:, n_known_classes:], labels_onehot[:, n_known_classes:])

            # Compute loss based on output and ground truth
            #loss = criterion(outputs, labels)

            if n_known_classes > 0:
                previous_outputs = torch.sigmoid(previous_model(images))

                # Compute the teaching loss and the distillation loss
                criterion_diss = nn.BCEWithLogitsLoss(reduction='mean')
                distil_loss = criterion_diss(outputs[:, :n_known_classes], previous_outputs[:, :n_known_classes])
                loss += distil_loss
                """
                prev_softmax = F.softmax(previous_outputs, dim=1)
                log_softmax = F.log_softmax(outputs[:, :n_known_classes], dim=1)
                S = torch.sum(prev_softmax * log_softmax, axis=1)

                loss -= torch.mean(S, axis=0).data
                """

            if current_step % Config.LOG_FREQUENCY == 0:
                # Log the information and add to tensorboard
                with torch.no_grad():
                    _, preds = torch.max(outputs, 1)
                    accuracy = torch.sum(preds == labels) / float(len(labels))

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                          .format(epoch + 1, current_step, loss.item(), accuracy.item()))

                    if log_dir is not None:
                        tb_writer.add_scalar('loss', loss.item(), current_step)
                        tb_writer.add_scalar('accuracy', accuracy.item(), current_step)

            # Compute gradients for each layer and update weights
            loss.backward()  # backward pass: computes gradients
            optimizer.step()  # update weights based on accumulated gradients

            current_step += 1

        # Step the scheduler
        scheduler.step()

    if log_dir is not None:
        tb_writer.close()
