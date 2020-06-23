from .resnet import resnet20, resnet32, resnet56
from .iCaRL import iCaRL
from .lwf import LwF
from .simple_learner import SimpleLearner
from example_usage.test_icarl_extended import iCaRLExtended

__all__ = ('resnet20', 'resnet32', 'resnet56',
           'iCaRL', 'LwF', 'SimpleLearner', 'iCaRLExtended')
