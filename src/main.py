import os

import torch

import data
import loss
import model
import utility
from option import args
from trainer import Trainer

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
checkpoint = utility.checkpoint(args)


def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.validate()
            print("interference")
            _model.model.switch_to_deploy()
            t.test()

            checkpoint.done()


if __name__ == '__main__':
    main()
