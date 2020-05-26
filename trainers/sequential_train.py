import os
from datetime import datetime

import torch

from trainers.train_once import train_model, test_model


def sequential_train(net, split_datasets, criterion, optimizer_factory,
                     scheduler_factory, batch_size=128, num_epochs=10,
                     log_dir_prefix=None):
    """ This function trains a net with incremental learning applied to the
        split dataset, using a basic procedure. For each iteration over the
        split dataset, it will train the network on the training dataset,
        and then the network will be evaluated over the test set.

    Args:
        net: The network to train.
        split_datasets(iterable): A iterator with a train and test datasets
            split into groups.
        criterion: The selected loss function.
        optimizer_factory: A object that implements the create_optimizer
            method which returns the network's optimizer.
        scheduler_factory: A object that implements the create_scheduler
            method which returns the scheduler to adapt the learning rate.
        num_epochs (int): The number of epochs for training each group.
        batch_size (int): The data bath size.
        log_dir_prefix (string, optional): The path to a folder to save the
            logs of training accuracy and training loss with tensorboard.
            If None, it will no log.
    """

    if log_dir_prefix is not None:
        now = datetime.now()
        log_dir_prefix = os.path.join(log_dir_prefix, now.strftime('%m-%d %H:%M:%S'))

    for idx, (train_dataset, test_dataset) in enumerate(split_datasets):
        print('\nGroup {}/{}. Training on classes: {}'.format(idx, split_datasets.get_total_groups(),
                                                              split_datasets.get_train_groups_classes()[idx]))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=4)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=4)
        if log_dir_prefix is not None:
            log_dir_prefix = os.path.join(log_dir_prefix, 'group_' + str(idx))

        optimizer = optimizer_factory.create_optimizer()
        scheduler = scheduler_factory.create_scheduler(optimizer)

        train_model(net, train_dataloader, criterion, optimizer, scheduler, num_epochs, log_dir_prefix)

        test_model(net, test_dataloader)
