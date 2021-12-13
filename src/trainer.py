from decimal import Decimal

import torch
import torch.nn.utils as utils
import utility
import wandb
import numpy as np
from tqdm import tqdm
from pytorch_msssim import ssim


def init_wandb_logging(args):
    if not args.wandb_disable:
        wandb.init(project=args.wandb_project_name, entity="midl21t1")
        wandb.config.update(args)



def add_test_wandb_logs(args, dataset_to_scale_to_sum_losses, dataset_to_scale_to_sum_psnr, dataset_to_scale_to_sum_ssim, mean_time_forward_pass, step_name, epoch):
    if args.wandb_disable:
        return
    test_logs = {}
    test_mean_loss = 0

    def add_to_testlog(metric, metric_dict):
        for dataset, values_by_scale in metric_dict.items():
            for scale, sum_metric in values_by_scale.items():
                test_logs[f'{dataset}_{metric}_scale_{scale}'] = sum_loss

    add_to_testlog("loss", dataset_to_scale_to_sum_losses)
    add_to_testlog("psnr", dataset_to_scale_to_sum_psnr)
    add_to_testlog("ssim", dataset_to_scale_to_sum_ssim)
    
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

        dataset_to_scale_to_sum_losses = {dataset.dataset.name: {scale: 0 for scale in self.scale} for dataset in loader}
        dataset_to_scale_to_sum_psnr = {dataset.dataset.name: {scale: 0 for scale in self.scale} for dataset in loader}
        dataset_to_scale_to_sum_ssim = {dataset.dataset.name: {scale: 0 for scale in self.scale} for dataset in loader}

        for idx_data, d in enumerate(loader):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    timer_test.tic()
                    sr = self.model(lr, idx_scale)
                    time_diff = timer_test.toc()
                    time_diff /= hr.shape[0]

                    durations.append(time_diff)
                    loss = self.loss(sr, hr)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    psnr = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    my_ssim = ssim(sr, hr, data_range=self.args.rgb_range)
                    self.ckp.log[-1, idx_data, idx_scale] += psnr
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                    dataset_to_scale_to_sum_losses[d.dataset.name][scale] += loss
                    dataset_to_scale_to_sum_psnr[d.dataset.name][scale] += psnr
                    dataset_to_scale_to_sum_ssim[d.dataset.name][scale] += my_ssim

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                dataset_to_scale_to_sum_losses[d.dataset.name][scale] /= len(d)
                dataset_to_scale_to_sum_psnr[d.dataset.name][scale] /= len(d)
                dataset_to_scale_to_sum_ssim[d.dataset.name][scale] /= len(d)


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
                self.ckp.write_log(
                    '[{} x{}]\tSSIM: {:.3f}'.format(
                        d.dataset.name,
                        scale,
                       dataset_to_scale_to_sum_ssim[d.dataset.name][scale],
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

    
        add_test_wandb_logs(self.args, dataset_to_scale_to_sum_losses, dataset_to_scale_to_sum_psnr, dataset_to_scale_to_sum_ssim, mean_time_forward_pass, step_name, epoch)

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
            return epoch > self.args.epochs
