import glob
import os

import numpy as np
import torch
import torch.utils.data as data

from data import common, srdata


class EvaluationDataset(srdata.SRData):
    def __init__(self, args, name='', train=False, benchmark=True):
        super(EvaluationDataset, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'hr')
        self.dir_lr = os.path.join(self.apath, 'lr')
        self.ext = ('.png', '.png')

    def _get_lr_filepath(self, hr_filename, scale):
        lr_filename = '_'.join(
            hr_filename.split('_')[
                :3]) + f'_{scale}_LR{self.ext[1]}'
        return os.path.join(
            self.dir_lr, f'x{scale}', f'{lr_filename}'
        )
