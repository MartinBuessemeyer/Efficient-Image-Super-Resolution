from decimal import Decimal

import torch
import torch.nn.utils as utils
import utility
import wandb
import numpy as np
from tqdm import tqdm


def init_wandb_logging(args):
    if not args.wandb_disable:
        wandb.init(project=args.wandb_project_name, entity="midl21t1")
        wandb.config.update(args)


def add_test_wandb_logs(args, num_files, scale_to_sum_losses, scale_to_sum_psnr, mean_time_forward_pass, step_name, epoch):
    if args.wandb_disable:
        return
    test_logs = {}
    test_mean_loss = 0
    for scale, sum_loss in scale_to_sum_losses.items():
        test_mean_loss += sum_loss / num_files
        test_logs[f'loss_scale_{scale}'] = sum_loss / num_files
    test_mean_loss /= len(scale_to_sum_losses.values())
    test_mean_psnr = 0
    for scale, sum_psnr in scale_to_sum_psnr.items():
        test_mean_psnr += sum_psnr / num_files
        test_logs[f'psnr_scale_{scale}'] = sum_psnr / num_files
    test_mean_psnr /= len(scale_to_sum_psnr.values())
    test_logs['mean_loss'] = test_mean_loss
    test_logs['mean_psnr'] = test_mean_psnr
    test_logs['mean_forward_pass_time'] = mean_time_forward_pass
    wandb.log({step_name: test_logs}, step=epoch)


class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_validate = loader.loader_validate
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        init_wandb_logging(args)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        sum_loss = 0
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

            sum_loss += loss

        wandb.log({'train': {'loss': sum_loss / len(self.loader_train),
                             'lr': self.optimizer.get_lr()}})
        wandb.watch(self.model)

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test_or_validate(self, loader, step_name):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log(f'\n{step_name}:')
        self.ckp.add_log(
            torch.zeros(1, len(loader), len(self.scale))
        )
        self.model.eval()

        durations = []
        timer_test = utility.timer()
        if self.args.save_results:
            self.ckp.begin_background()

        scale_to_sum_losses = {scale: 0 for scale in self.scale}
        scale_to_sum_psnr = {scale: 0 for scale in self.scale}
        num_files = 0

        for idx_data, d in enumerate(loader):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    timer_test.tic()
                    sr = self.model(lr, idx_scale)
                    time_diff = timer_test.toc()
                    durations.append(time_diff)
                    loss = self.loss(sr, hr)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    psnr = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    self.ckp.log[-1, idx_data, idx_scale] += psnr
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                    scale_to_sum_losses[scale] += loss
                    scale_to_sum_psnr[scale] += psnr
                    num_files += 1

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
        mean_time_forward_pass = np.mean(durations)
        self.ckp.write_log('Mean time forward pass: {:.5f}s\n'.format(mean_time_forward_pass))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        add_test_wandb_logs(self.args, num_files, scale_to_sum_losses, scale_to_sum_psnr, mean_time_forward_pass, step_name, epoch)

        torch.set_grad_enabled(True)

    def validate(self):
        self.test_or_validate(self.loader_validate, 'validate')

    def test(self):
        self.test_or_validate(self.loader_test, 'test')

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.validate()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
