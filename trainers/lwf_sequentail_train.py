import copy
import os
from datetime import datetime

from torch import nn
from torch.utils.data import DataLoader

from trainers.lwf_train import lwf_train_model
from trainers.train_once import test_model


def lwf_sequential_train(net, split_datasets, criterion, optimizer_factory,
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

    n_known_classes = 0
    previous_model = None

    for idx, (train_dataset, test_dataset) in enumerate(split_datasets):
        print('\nGroup {}/{}. Training on classes: {}'.format(idx, split_datasets.get_total_groups(),
                                                              split_datasets.get_train_groups_classes()[idx]))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        if log_dir_prefix is not None:
            log_dir_prefix = os.path.join(log_dir_prefix, 'group_' + str(idx))

        optimizer = optimizer_factory.create_optimizer(net)
        scheduler = scheduler_factory.create_scheduler(optimizer)

        lwf_train_model(net, train_dataloader, criterion, optimizer, scheduler, num_epochs, n_known_classes, log_dir_prefix, previous_model)

        test_model(net, test_dataloader)

        n_known_classes += split_datasets.get_total_groups()

        previous_model = copy.deepcopy(net)

        # Update network fc layer with more outputs
        in_features = net.fc.in_features
        out_features = net.fc.out_features
        weight = net.fc.weight.data
        new_out_features = n_known_classes+split_datasets.get_total_groups()
        net.fc = nn.Linear(in_features, new_out_features)
        net.fc.weight.data[:out_features] = weight
