from abc import abstractmethod, ABC

from torch import optim


class OptimizerFactory(ABC):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def create_optimizer(self, net):
        pass


class SDGOptimizerAllFactory(OptimizerFactory):

    def create_optimizer(self, net):
        return optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), *self.args, **self.kwargs)

