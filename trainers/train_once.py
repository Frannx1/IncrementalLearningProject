import os

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_model(net, train_dataloader, optimizer, criterion, scheduler, num_epochs, log_dir=None):
    # By default, everything is loaded to cpu
    net = net.to(DEVICE)  # this will bring the network to GPU if DEVICE is cuda

    cudnn.benchmark  # Calling this optimizes runtime

    if log_dir is not None:
        # TensorboardX summary writer
        params_save = 'lr: {}, batch: {}, epochs: {}'.format(
            scheduler.get_last_lr()[0], train_dataloader.batch_size,
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
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            net.train()  # Sets module in training mode

            # PyTorch, by default, accumulates gradients after each backward pass
            # We need to manually set the gradients to zero before starting a new iteration
            optimizer.zero_grad()  # Zero-ing the gradients

            # Forward pass to the network
            outputs = net(images)

            # Compute loss based on output and ground truth
            loss = criterion(outputs, labels)

            # Log the information and add to tensorboard
            if current_step % LOG_FREQUENCY == 0:
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


def test_model(net, test_dataloader):
    net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
    net.train(False) # Set Network to evaluation mode

    running_corrects = 0
    for images, labels in tqdm(test_dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward Pass
        class_pred = net(images)

        # Get predictions
        _, preds = torch.max(class_pred.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / float(len(test_dataloader.dataset))

    print('\nTest Accuracy: {}'.format(accuracy))
    return accuracy

