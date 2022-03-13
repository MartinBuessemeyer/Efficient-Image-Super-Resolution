from decimal import Decimal

import numpy as np
import torch
import torchvision.transforms
import wandb
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

import utility as utility


def num_params_of_model(model):
    return sum((param.numel() for param in model.parameters()))


def init_wandb_logging(args, ckp):
    if not args.wandb_disable:
        wandb.init(project=args.wandb_project_name, entity="midl21t1")
        wandb.config.update(args)
        for key, val in wandb.config.items():
            ckp.add_csv_result(f'config.{key}', val, 1)


def add_test_wandb_logs(
        args,
        ckp,
        dataset_to_scale_to_sum_losses,
        dataset_to_scale_to_sum_psnr,
        dataset_to_scale_to_sum_ssim,
        mean_time_forward_pass,
        step_name,
        epoch,
        test_csv_log_length):
    if args.wandb_disable:
        return
    test_logs = {dataset: {} for dataset in dataset_to_scale_to_sum_psnr.keys()}

    def add_to_testlog(metric, metric_dict):
        for dataset, values_by_scale in metric_dict.items():
            for scale, sum_metric in values_by_scale.items():
                metric_key = f'{metric}_scale_{scale}'
                ckp.add_csv_result(f'{step_name}.{dataset}.{metric_key}', sum_metric,
                                   epoch if test_csv_log_length else None)
                test_logs[dataset][metric_key] = sum_metric

    add_to_testlog("loss", dataset_to_scale_to_sum_losses)
    add_to_testlog("psnr", dataset_to_scale_to_sum_psnr)
    add_to_testlog("ssim", dataset_to_scale_to_sum_ssim)

    test_logs[f'{step_name}.mean_forward_pass_time'] = mean_time_forward_pass
    ckp.add_csv_result(f'{step_name}.mean_forward_pass_time', mean_time_forward_pass,
                       epoch if test_csv_log_length else None)
    wandb.log({step_name: test_logs}, step=epoch)


class Trainer:
    def __init__(self, args, loader, my_model, my_loss, ckp, pruning_scheduler):
        self.args = args
        init_wandb_logging(args, ckp)

        self.scale = args.scale

        self.ckp = ckp
        self.pruning_scheduler = pruning_scheduler
        self.loader_train = loader.loader_train
        self.loader_validate = loader.loader_validate
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

    def train(self):
        epoch = self.optimizer.get_last_epoch() + 1

        if self.pruning_scheduler.should_prune():
            prev_layer_size, new_layer_size = self.model.model.prune()
            self.model.model.to(self.device)
            self.ckp.write_log(f'[Epoch {epoch}]\tPruning model from layer size {prev_layer_size} to {new_layer_size}')

        self.loss.step()
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.ckp.add_csv_result('Epoch', epoch, epoch)
        self.ckp.add_csv_result('lr', lr, epoch)
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        pass_timer = utility.timer()
        durations = []
        # TEMP
        self.loader_train.dataset.set_scale(0)
        sum_loss = 0
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            pass_timer.tic()
            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            durations.append(pass_timer.toc() / lr.size()[0])
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
        if not self.args.wandb_disable:
            mean_loss = sum_loss / len(self.loader_train)
            num_parameters = num_params_of_model(self.model.model)
            mean_train_duration = np.mean(durations)
            print(f'Mean Train Duration: {mean_train_duration}')
            wandb.log({'train': {'loss': mean_loss,
                                 'lr': self.optimizer.get_lr(),
                                 'time': mean_train_duration},
                       'num_parameters': num_parameters})
            self.ckp.add_csv_result('train.loss', mean_loss, epoch)
            self.ckp.add_csv_result('num_parameters', num_parameters, epoch)
            self.ckp.add_csv_result('train.time', mean_train_duration, epoch)

        self.loss.end_log(len(self.loader_train))
        self.optimizer.schedule()

    def get_averaged_forward_pass_time(self, loader, num_iters_to_avg=25):
        torch.set_grad_enabled(False)
        self.model.eval()

        durations = []
        timer_test = utility.timer()
        for i in range(num_iters_to_avg):
            for idx_data, d in enumerate(loader):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    for lr, hr, filename in tqdm(d, ncols=80):
                        batch_size = lr.size()[0]
                        lr, hr = self.prepare(lr, hr)
                        timer_test.tic()
                        _ = self.model(lr, idx_scale)
                        time_diff = timer_test.toc()
                        time_diff /= batch_size
                        durations.append(time_diff)
        mean_time_forward_pass = float(np.mean(durations))
        torch.set_grad_enabled(True)
        return mean_time_forward_pass

    def test_or_validate(self, loader, step_name, test_csv_log_length=False, save_model=True):
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

        dataset_to_scale_to_sum_losses = {dataset.dataset.name: {
            scale: 0 for scale in self.scale} for dataset in loader}
        dataset_to_scale_to_sum_psnr = {dataset.dataset.name: {
            scale: 0 for scale in self.scale} for dataset in loader}
        dataset_to_scale_to_sum_ssim = {dataset.dataset.name: {
            scale: 0 for scale in self.scale} for dataset in loader}

        for idx_data, d in enumerate(loader):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    batch_size = lr.size()[0]
                    lr, hr = self.prepare(lr, hr)
                    timer_test.tic()
                    sr = self.model(lr, idx_scale)
                    time_diff = timer_test.toc()
                    time_diff /= batch_size

                    durations.append(time_diff)
                    # handle hr wrong resolution
                    if sr.shape != hr.shape:
                        h, w = sr.shape[2:]
                        hr = torchvision.transforms.functional.resize(hr, size=(h, w),
                                                                      interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC)
                    loss = self.loss(sr, hr)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]

                    psnr, ssim = self.calculate_batch_ssim_psnr(batch_size, hr, sr)

                    self.ckp.log[-1, idx_data, idx_scale] += psnr
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results and save_model:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                    dataset_to_scale_to_sum_losses[d.dataset.name][scale] += loss
                    dataset_to_scale_to_sum_psnr[d.dataset.name][scale] += psnr
                    dataset_to_scale_to_sum_ssim[d.dataset.name][scale] += ssim

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
        self.ckp.write_log(
            'Mean time forward pass: {:.5f}s\n'.format(mean_time_forward_pass))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if (not self.args.test_only) and save_model:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        add_test_wandb_logs(
            self.args,
            self.ckp,
            dataset_to_scale_to_sum_losses,
            dataset_to_scale_to_sum_psnr,
            dataset_to_scale_to_sum_ssim,
            mean_time_forward_pass,
            step_name,
            epoch, test_csv_log_length)
        torch.set_grad_enabled(True)

    def calculate_batch_ssim_psnr(self, batch_size, hr, sr):
        psnr = 0
        ssim = 0
        for batch_idx in range(batch_size):
            sr_numpy = sr[batch_idx, ...].detach().cpu().numpy()
            hr_numpy = hr[batch_idx, ...].detach().cpu().numpy()
            ssim += structural_similarity(sr_numpy,
                                          hr_numpy,
                                          channel_axis=0,
                                          data_range=self.args.rgb_range)
            psnr += peak_signal_noise_ratio(
                sr_numpy, hr_numpy, data_range=self.args.rgb_range)
        ssim /= batch_size
        psnr /= batch_size
        return psnr, ssim

    def validate(self):
        self.test_or_validate(self.loader_validate, 'validate', True)

    def test(self):
        self.test_or_validate(self.loader_test, 'test', save_model=False)
        num_parameters = num_params_of_model(self.model.model)
        mean_inference_time = self.get_averaged_forward_pass_time(self.loader_test)
        self.ckp.write_log(
            'Averaged mean inference time forward pass: {:.5f}s\n'.format(mean_inference_time))
        if not self.args.wandb_disable:
            self.ckp.add_csv_result('num_parameters_production', num_parameters)
            self.ckp.add_csv_result('avg_inference_forward_pass_time', mean_inference_time)
            wandb.log({'num_parameters_production': num_parameters})
            wandb.log({'avg_inference_forward_pass_time': mean_inference_time})

    def prepare(self, *args):

        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.validate()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch > self.args.epochs
