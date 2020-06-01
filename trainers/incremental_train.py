import os
from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def incremental_train(incremental_learner, split_datasets, optimizer_factory,
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

    log_dir = None
    if log_dir_prefix is not None:
        now = datetime.now()
        log_dir_prefix = os.path.join(log_dir_prefix, now.strftime('%m-%d %H:%M:%S'))
        tb_writer = SummaryWriter(log_dir=log_dir_prefix)

    for idx, (train_dataset, test_dataset) in enumerate(split_datasets):
        print('\nGroup {}/{}. Training on classes: {}'.format(idx, split_datasets.get_total_groups(),
                                                              split_datasets.get_train_groups_classes()[idx]))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        if log_dir_prefix is not None:
            log_dir = os.path.join(log_dir_prefix, 'group_' + str(idx))

        optimizer = optimizer_factory.create_optimizer(incremental_learner)
        scheduler = scheduler_factory.create_scheduler(optimizer)

        # Train on the current group
        incremental_learner.before_task(train_dataloader)

        incremental_learner.train_task(train_dataloader, optimizer, scheduler, num_epochs, log_dir=log_dir)

        incremental_learner.after_task(train_dataloader)

        # Evaluate on the groups seen up to this iteration
        test_acc = incremental_learner.eval_task(test_dataloader)

        if log_dir_prefix is not None:
            # Log the test accuracy after training a group
            tb_writer.add_scalar('test accuracy', float(test_acc), idx)
