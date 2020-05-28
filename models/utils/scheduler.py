from abc import abstractmethod

from torch import optim


class SchedulerFactory(object):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def create_scheduler(self, optimizer):
        pass


class StepLRSchedulerFactory(SchedulerFactory):

    def create_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, *self.args,
                                         **self.kwargs)


class MultiStepLRSchedulerFactory(SchedulerFactory):

    def create_scheduler(self, optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer,
                                              *self.args,
                                              **self.kwargs)
