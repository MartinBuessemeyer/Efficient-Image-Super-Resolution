import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=False, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'hr')
        self.dir_lr = os.path.join(self.apath, 'lr')
        self.ext = ('.png', '.png')

    def _get_lr_filepath(self, hr_filename, scale):
        return os.path.join(
                    self.dir_lr, f'x{scale}', f'{hr_filename.replace("HR", "LR")}{self.ext[1]}'
                )
