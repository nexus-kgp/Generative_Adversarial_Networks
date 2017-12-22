import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np

from torch.autograd import Variable
from torch import optim
from model import G12, G21
from model import D1, D2


class CycleGAN(object):
	def __init__(self, config, svhn_loader, mnist_loader):