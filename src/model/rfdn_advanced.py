import model.block_advanced as B
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from model import common
from model.block_advanced import conv_layer, SRB


def make_model(args, parent=False):
    model = RFDNAdvanced(args)
    return model

def _merge_identity_mask(current_mask, new_mask, device):
    combined_mask = []
    new_mask_idx_offset = 0
    for idx in range(len(current_mask)):
        if not current_mask[idx]:
            new_mask_idx_offset += 1
            combined_mask.append(False)
        else:
            combined_mask.append(new_mask[idx + new_mask_idx_offset])
    return torch.tensor(combined_mask, device=device)

def _get_new_parameter(new_weight_or_bias, device):
    return torch.nn.Parameter(new_weight_or_bias.to(device=device))


def _get_mask_for_pruning(srb):
    eval_conv = srb.get_equivalent_conv_layer()
    eval_conv = prune.ln_structured(eval_conv, "weight", amount=0.1, n=1, dim=0)
    mask = (eval_conv.weight.sum(dim=(1, 2, 3)) != 0)
    num_filters_remaining = int(torch.sum(mask)
    return mask, num_filters_remaining


def _get_pruned_subsequent_srb_block(srb, mask, remaining_in_channels, device):
    new_srb = SRB(remaining_in_channels, srb.out_channels, srb.activation, srb.deploy)
    for conv, new_conv in zip([srb.conv3.conv, srb.conv1.conv],
                              [new_srb.conv3.conv, srb.conv1.conv]):
        new_conv.weight = _get_new_parameter(conv.weight[:, mask, ...], device)
        new_conv.bias = conv.bias
    return new_srb.to(device)


# Applies the pruning mask to the given convolution layer by treating it as
# the layer that is pruned.
def _get_pruned_srb_by_mask(srb, mask, remaining_filters, device):
    assert not srb.deploy
    conv = srb.get_equivalent_conv_layer()
    num_filters, num_input_channels, h, w = conv.weight.shape
    assert num_filters == len(mask)
    new_srb = SRB(num_input_channels, remaining_filters, srb.activation, srb.deploy)
    new_srb.conv3.conv.weight = _get_new_parameter(conv.weight[mask], device)
    new_srb.conv3.conv.bias = _get_new_parameter(conv.bias[mask], device)
    return new_srb.to(device)


# Applies the pruning mask to the given convolution layer by treating it
# as a layer that comes after the pruned layer.
def _get_pruned_subsequent_distilled_layer_by_mask(conv, mask, remaining_in_channels, device):
    num_filters, num_input_channels, h, w = conv.weight.shape
    assert num_input_channels == len(mask)
    new_layer = conv_layer(remaining_in_channels, num_filters, h)
    new_layer.weight = _get_new_parameter(conv.weight[:, mask, ...], device)
    new_layer.bias = conv.bias
    return new_layer


class RFDNAdvanced(nn.Module):
    def __init__(self, args):
        super(RFDNAdvanced, self).__init__()

        if args.n_feats != 50:
            print(
                f'WARNING: Using non paper num output channels of {args.n_feats} instead of 50.')
        self.fea_conv = B.conv_layer(
            args.n_colors, args.n_feats, kernel_size=3)

        self.device = torch.device('cpu' if args.cpu else 'cuda')

        num_rfdb_blocks = 4
        self.B1 = B.RFDB(in_channels=args.n_feats)
        self.B2 = B.RFDB(in_channels=args.n_feats)
        self.B3 = B.RFDB(in_channels=args.n_feats)
        self.B4 = B.RFDB(in_channels=args.n_feats)
        self.c = B.conv_block(args.n_feats * num_rfdb_blocks,
                              args.n_feats, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(args.n_feats, args.n_feats, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(
            args.n_feats, args.n_colors, upscale_factor=args.scale[0])
        self.scale_idx = 0

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

    def forward(self, input):
        input = self.sub_mean(input)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        output = self.add_mean(output)
        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    def switch_to_deploy(self):
        self.B1.switch_to_deploy()
        self.B2.switch_to_deploy()
        self.B3.switch_to_deploy()
        self.B4.switch_to_deploy()

    def prune(self):
        for block in [self.B1, self.B2, self.B3, self.B4]:
            for srb_idx in range(len(block.srbs)):
                srb = block.srbs[srb_idx]
                mask, num_filters_remaining = _get_mask_for_pruning(srb)
                block.srbs[srb_idx] = _get_pruned_srb_by_mask(srb, mask, num_filters_remaining, self.device)
                if srb_idx == 0:
                    block.srbs[srb_idx].identity_mask = _merge_identity_mask(srb.identity_mask, mask, self.device)
                block.distilled_layers[srb_idx + 1] = _get_pruned_subsequent_distilled_layer_by_mask(
                    block.distilled_layers[srb_idx + 1], mask, num_filters_remaining, self.device)
                if srb_idx < len(block.srbs) - 1:
                    block.srbs[srb_idx + 1] = _get_pruned_subsequent_srb_block(block.srbs[srb_idx + 1],
                                                                               mask, num_filters_remaining, self.device)
