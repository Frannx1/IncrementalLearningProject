from abc import abstractmethod

from torch import nn


class MultiTaskLearner(nn.Module):
    """Base incremental learner.
    Methods are called in this order (& repeated for each new task):
    1. before_task
    2. train_task
    3. after_task
    4. eval_task

    Reference: https://github.com/AfricanxAdmiral/icarl/blob/master/inclearn/models/base.py
    """

    def before_task(self, train_loader, val_loader):
        pass

    @abstractmethod
    def train_task(self, train_loader, optimizer, scheduler, num_epochs, val_loader=None, log_dir=None):
        pass

    def after_task(self, train_loader):
        pass

    @abstractmethod
    def eval_task(self, eval_loader):
        pass

