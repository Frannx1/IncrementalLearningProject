from abc import abstractmethod

from torch import optim


class OptimizerFactory(object):

    @abstractmethod
    def create_optimizer(self):
        pass


class SDGOptimizerFactory(OptimizerFactory):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def create_optimizer(self):
        return optim.SGD(*self.args, **self.kwargs)

