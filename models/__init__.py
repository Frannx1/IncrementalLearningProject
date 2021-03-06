from .resnet import resnet20, resnet32, resnet56
from .iCaRL import iCaRL
from .lwf import LwF
from .simple_learner import SimpleLearner
from .iCaRL_extended import iCaRLExtended
from .iCaRL_selection import iCaRLSelection

__all__ = ('resnet20', 'resnet32', 'resnet56',
           'iCaRL', 'LwF', 'SimpleLearner', 'iCaRLExtended', 'iCaRLSelection')
