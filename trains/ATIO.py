"""
AIO -- All Trains in One
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

from trains import *

__all__ = ['ATIO']

TRAIN_MAP = {
    'self_mm': SELF_MM
}

class ATIO():
    def __init__(self):
        pass
    
    def getTrain(self, args):
        return TRAIN_MAP[args.modelName.lower()](args)
