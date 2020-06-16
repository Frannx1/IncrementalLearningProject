import copy
import os
from datetime import datetime

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from trainers.lwf_train import lwf_train_model
from trainers.train_once import test_model


def lwf_sequential_train(net, split_datasets, criterion, optimizer_factory,
                         scheduler_factory, batch_size=128, num_epochs=10,
                         log_dir_prefix=None):
    """ This function trains a net with incremental learning applied to the
        split dataset, using a learning without forgetting strategy.
        For each iteration over the split dataset, it will train the
        network on the training dataset, and then the network will be
        evaluated over the test set.
        Implementation of Learning without Forgetting paper
        https://arxiv.org/pdf/1606.09282.pdf.
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
        tb_writer = SummaryWriter(log_dir=log_dir_prefix)

    n_known_classes = 0
    n_classes = net.fc.out_features
    previous_model = None

    log_dir = None
    for idx, (train_dataset, test_dataset) in enumerate(split_datasets):
        print('\nGroup {}/{}. Training on classes: {}'.format(idx+1, split_datasets.get_total_groups(),
                                                              split_datasets.get_train_groups_classes()[idx]))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        if log_dir_prefix is not None:
            log_dir = os.path.join(log_dir_prefix, 'group_' + str(idx))

        if n_known_classes == n_classes:
            # Update network fc layer with more outputs
            n = len(set(train_dataloader.dataset.targets))
            n_classes += n
            in_features = net.fc.in_features
            out_features = net.fc.out_features
            weight = net.fc.weight.data
            net.fc = nn.Linear(in_features, n_classes)
            net.fc.weight.data[:out_features] = weight

        optimizer = optimizer_factory.create_optimizer(net)
        scheduler = scheduler_factory.create_scheduler(optimizer)

        # Train on the current group
        lwf_train_model(net, train_dataloader, criterion, optimizer, scheduler, num_epochs, n_known_classes,
                        log_dir, previous_model)

        # Evaluate on the groups seen up to this iteration
        test_acc = test_model(net, test_dataloader)

        if log_dir_prefix is not None:
            # Log the test accuracy after training a group
            tb_writer.add_scalar('test accuracy', float(test_acc), idx)

        n_known_classes = n_classes

        previous_model = copy.deepcopy(net)
