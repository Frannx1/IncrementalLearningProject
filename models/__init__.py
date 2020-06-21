from .resnet import resnet20, resnet32, resnet56
from .iCaRL import iCaRL
from .lwf import LwF
from .simple_learner import SimpleLearner

__all__ = ('resnet20', 'resnet32', 'resnet56',
           'iCaRL', 'LwF', 'SimpleLearner')
