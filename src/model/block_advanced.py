from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=padding,
        bias=bias,
        dilation=dilation,
        groups=groups)


def conv_bn(in_channels, out_channels, kernel_size, disable_batchnorm=False):
    result = nn.Sequential()
    if not disable_batchnorm:
        result.add_module(
        'conv',
        conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            bias=False))
        result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    else:
        result.add_module(
            'conv',
            conv_layer(
                in_channels,
                out_channels,
                kernel_size))
    return result


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(
            'normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(
            'padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(
        in_nc,
        out_nc,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        pad_type='zero',
        norm_type=None,
        act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2,
                                                              keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class SRB(nn.Module):
    def __init__(self, in_channels, out_channels, activation, deploy=False, disable_batchnorm=False):
        super(SRB, self).__init__()

        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        self.identity_mask = torch.ones(out_channels, dtype=torch.bool)
        self.disable_batchnorm = disable_batchnorm

        if self.deploy:
            self.reparam = conv_layer(in_channels, out_channels, 3)
        else:
            self.conv3 = conv_bn(in_channels, out_channels, 3, disable_batchnorm=disable_batchnorm)
            self.conv1 = conv_bn(in_channels, out_channels, 1, disable_batchnorm=disable_batchnorm)
            self.identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and not disable_batchnorm else torch.nn.Identity()

    def get_layer_size(self):
        return self.reparam.out_channels if self.deploy else self.conv1.conv.out_channels

    def forward(self, input):
        if self.deploy:
            residual = self.activation(self.reparam(input))
        else:
            conv3 = (self.conv3(input))
            conv1 = (self.conv1(input))
            residual = self.activation(conv3 + conv1 + self.identity(input[:, self.identity_mask, ...]))

        return residual

    def get_equivalent_conv_layer(self):
        kernel_3x3, bias_3x3 = self._fuse_bn_tensor(self.conv3)
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.conv1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.identity)
        res_kernel = kernel_3x3 + self.pad_1x1_to_3x3_tensor(kernel_1x1) + kernel_id
        res_bias = bias_3x3 + bias_1x1 + bias_id
        res_conv = conv_layer(self.in_channels, self.out_channels, 3)
        res_conv.weight = torch.nn.Parameter(res_kernel)
        res_conv.bias = torch.nn.Parameter(res_bias)
        return res_conv

    def pad_1x1_to_3x3_tensor(self, kernel_1x1):
        if kernel_1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel_1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None or isinstance(branch, nn.Identity):
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            bias = branch.conv.bias if hasattr(branch.conv, 'bias') and branch.conv.bias is not None else 0
            if hasattr(branch, 'bn'):
                running_mean = branch.bn.running_mean
                running_var = branch.bn.running_var
                gamma = branch.bn.weight
                beta = branch.bn.bias
                eps = branch.bn.eps
            else:
                # results in 0 as bias and * 1 in kernel
                running_mean = torch.tensor(0, device=kernel.device)
                running_var = torch.tensor(1, device=kernel.device)
                gamma = torch.tensor(1, device=kernel.device)
                beta = torch.tensor(0, device=kernel.device)
                eps = torch.tensor(0, device=kernel.device)
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(
                    kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            bias = torch.tensor(0, device=branch.weight.device)
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, bias + beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'reparam'):
            return
        self.reparam = self.get_equivalent_conv_layer()
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv3')
        self.__delattr__('conv1')
        if hasattr(self, 'identity'):
            self.__delattr__('identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class RFDB(nn.Module):
    def __init__(self, in_channels, disable_batchnorm=False):
        super(RFDB, self).__init__()
        self.distilled_channels = in_channels // 2
        self.remaining_channels = in_channels
        self.activation = activation('lrelu', neg_slope=0.05)

        distilled1 = conv_layer(in_channels, self.distilled_channels, 1)
        distilled2 = conv_layer(
            self.remaining_channels, self.distilled_channels, 1)
        distilled3 = conv_layer(
            self.remaining_channels, self.distilled_channels, 1)
        distilled4 = conv_layer(
            self.remaining_channels, self.distilled_channels, 3)
        self.distilled_layers = nn.ModuleList([distilled1, distilled2, distilled3, distilled4])

        srb1 = SRB(in_channels, self.remaining_channels, self.activation, disable_batchnorm=disable_batchnorm)
        srb2 = SRB(
            self.remaining_channels, self.remaining_channels, self.activation, disable_batchnorm=disable_batchnorm)
        srb3 = SRB(
            self.remaining_channels, self.remaining_channels, self.activation, disable_batchnorm=disable_batchnorm)
        self.srbs = nn.ModuleList([srb1, srb2, srb3])

        self.distilled = conv_layer(
            self.distilled_channels * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled1 = self.activation(self.distilled_layers[0](input))
        srb1 = self.srbs[0](input)

        distilled2 = self.activation(self.distilled_layers[1](srb1))
        srb2 = self.srbs[1](srb1)

        distilled3 = self.activation(self.distilled_layers[2](srb2))
        srb3 = self.srbs[2](srb2)

        distilled4 = self.activation(self.distilled_layers[3](srb3))

        out = torch.cat(
            [distilled1, distilled2, distilled3, distilled4], dim=1)
        out_fused = self.esa(self.distilled(out))

        return out_fused

    def switch_to_deploy(self):
        for srb in self.srbs:
            srb.switch_to_deploy()


def pixelshuffle_block(
        in_channels,
        out_channels,
        upscale_factor=2,
        kernel_size=3,
        stride=1):
    conv = conv_layer(in_channels, out_channels *
                      (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)
