import os

import torch

import data
import loss
import model
import utility
from option import args
from pruning_scheduler import NoPrune, PruneAfterEpochs
from trainer import Trainer

# PyCharm remote debugging setup
if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    import pydevd_pycharm

    pydevd_pycharm.settrace(
        'localhost',
        port=int(
            os.environ.get(
                'REMOTE_PYCHARM_DEBUG_PORT',
                "12043")),
        stdoutToServer=True,
        stderrToServer=True,
        suspend=False)

torch.manual_seed(args.seed)
checkpoint = utility.Checkpoint(args)


def main():
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        _pruning_scheduler = NoPrune(None) if args.epochs_before_pruning == None else PruneAfterEpochs(
            args.epochs_before_pruning)
        t = Trainer(args, loader, _model, _loss, checkpoint, _pruning_scheduler)
        while not t.terminate():
            t.train()
            t.validate()
        print("interference")
        _model.model.switch_to_deploy()
        t.test()

        checkpoint.done()


if __name__ == '__main__':
    main()
