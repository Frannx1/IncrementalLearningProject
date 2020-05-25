import os
from datetime import datetime

import torch

from trainers.train_once import train_model, test_model


def basic_incremental_train(net, split_datasets, optimizer, criterion, scheduler, batch_size=128, num_epochs=10,
                            log_dir=None):
    """ This function trains a net with incremental learning applied to the
        split dataset, using a basic procedure. For each iteration over the
        split dataset, it will train the network on the training dataset,
        and then the network will be evaluated over the test set.

    Args:
        net: The network to train.
        split_datasets(iterable): A iterator with a train and test datasets
            split into groups.
        optimizer: The network's optimizer
        criterion: The selected loss function
        scheduler: The scheduler to adapt the learning rate.
        num_epochs (int): The number of epochs for training each group.
        batch_size (int): The data bath size.
        log_dir (string, optional): The path to a folder to save the logs of
            training accuracy and training loss with tensorboard. If None,
            it will no log.

    """

    if log_dir is not None:
        now = datetime.now()
        log_dir = os.path.join(log_dir, now.strftime('%m-%d %H:%M:%S'))

    for idx, train_dataset, test_dataset in enumerate(split_datasets):
        print('\nGroup {}/{}. Training on classes: {}'.format(idx, split_datasets.get_total_groups(),
                                                              split_datasets.get_train_groups_classes()[idx]))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=4)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=4)
        if log_dir is not None:
            log_dir = os.path.join(log_dir, 'group_' + str(idx))

        train_model(net, train_dataloader, optimizer, criterion, scheduler, num_epochs, log_dir)

        test_model(net, test_dataloader)
