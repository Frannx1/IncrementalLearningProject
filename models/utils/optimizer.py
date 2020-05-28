from abc import abstractmethod

from torch import optim


class OptimizerFactory(object):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def create_optimizer(self, net):
        pass


class SDGOptimizerAllFactory(OptimizerFactory):

    def create_optimizer(self, net):
        return optim.SGD(net.parameters(), *self.args, **self.kwargs)

