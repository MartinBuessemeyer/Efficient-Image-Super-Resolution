# Residual Feature Distillation Network for Lightweight Image Super-Resolution
# https://arxiv.org/abs/2009.11551
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import model.block as B


def make_model(args, parent=False):
    model = RFDN(args)
    return model


class RFDN(nn.Module):
    def __init__(self, args):
        super(RFDN, self).__init__()

        if args.n_feats != 50:
            print(
                f'WARNING: Using non paper num output channels of {args.n_feats} instead of 50.')
        self.fea_conv = B.conv_layer(
            args.n_colors, args.n_feats, kernel_size=3)

        num_rfdb_blocks = 4
        self.B1 = B.RFDB(in_channels=args.n_feats)
        self.B2 = B.RFDB(in_channels=args.n_feats)
        self.B3 = B.RFDB(in_channels=args.n_feats)
        self.B4 = B.RFDB(in_channels=args.n_feats)
        self.c = B.conv_block(
            args.n_feats *
            num_rfdb_blocks,
            args.n_feats,
            kernel_size=1,
            act_type='lrelu')

        self.LR_conv = B.conv_layer(args.n_feats, args.n_feats, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(
            args.n_feats,
            args.n_colors,
            upscale_factor=args.scale[0])
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    def switch_to_deploy(self):
        pass

    def prune(self):
        for block in [self.B1, self.B2, self.B3, self.B4]:
            for conv_layer in [block.c1_r, block.c2_r, block.c3_r]:
                prune.ln_structured(
                    conv_layer, "weight", amount=0.1, n=1, dim=0)
