import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.dir_hr = os.path.join(dir_data, 'hr')
        self.dir_lr = os.path.join(self.apath, 'lr')
        self.ext = ('', '.png')

