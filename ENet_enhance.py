#!/usr/bin/env python3.10
# MIT License (same header as your project)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------- Utils ----------------------

def random_weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight.data)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()


def conv_block(in_dim, out_dim, **kwconv):
    bias = kwconv.pop("bias", False)
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, bias=bias, **kwconv),
        nn.BatchNorm2d(out_dim),
        nn.PReLU()
    )



def conv_block_asym(in_dim, out_dim, *, kernel_size: int):
    pad = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size, 1), padding=(pad, 0), bias=False),
        nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size), padding=(0, pad), bias=False),
        nn.BatchNorm2d(out_dim),
        nn.PReLU()
    )


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, max(1, ch // r), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, ch // r), ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.fc(self.avg(x))
        return x * w


class BoundaryRefine(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class AttentionGate(nn.Module):
    def __init__(self, ch_x: int, ch_g: int):
        super().__init__()
        self.theta_x = nn.Conv2d(ch_x, ch_x, kernel_size=1, bias=False)
        self.phi_g   = nn.Conv2d(ch_g, ch_x, kernel_size=1, bias=False)
        self.bn      = nn.BatchNorm2d(ch_x)
        self.act     = nn.ReLU(inplace=True)
        self.psi     = nn.Conv2d(ch_x, ch_x, kernel_size=1, bias=True)
        self.sig     = nn.Sigmoid()

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        att = self.bn(self.theta_x(x) + self.phi_g(g))
        att = self.act(att)
        att = self.sig(self.psi(att))
        return x * att


# ---------------------- Bottlenecks ----------------------

class BottleNeck(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor,
                 *, dropoutRate=0.01, dilation=1,
                 asym: bool = False, dilate_last: bool = False,
                 use_se: bool = True):
        super().__init__()
        mid_dim: int = in_dim // projectionFactor

        self.block0 = conv_block(in_dim, mid_dim, kernel_size=1)
        if not asym:
            self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=dilation, dilation=dilation)
        else:
            self.block1 = conv_block_asym(mid_dim, mid_dim, kernel_size=5)
        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

        self.se = SEBlock(out_dim) if use_se else nn.Identity()
        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block(in_dim, out_dim, kernel_size=1)
        elif dilate_last:
            self.conv_out = conv_block(in_dim, out_dim, kernel_size=3, padding=1)
        else:
            self.conv_out = nn.Identity()

    def forward(self, in_) -> Tensor:
        b0 = self.block0(in_)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)
        do = self.se(do)
        output = self.PReLU_out(self.conv_out(in_) + do)
        return output


class BottleNeckDownSampling(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor):
        super().__init__()
        mid_dim: int = in_dim // projectionFactor

        self.maxpool0 = nn.MaxPool2d(2, return_indices=True)
        self.block0 = conv_block(in_dim, mid_dim, kernel_size=2, padding=0, stride=2)
        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)
        self.do = nn.Dropout(p=0.01)
        self.PReLU = nn.PReLU()

    def forward(self, in_) -> tuple[Tensor, Tensor]:
        maxpool_output, indices = self.maxpool0(in_)
        b0 = self.block0(in_)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        _, c, _, _ = maxpool_output.shape
        out = do
        out[:, :c, :, :] += maxpool_output
        final_output = self.PReLU(out)
        return final_output, indices


class BottleNeckUpSampling(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor, *, skip_ch: int = None, att_gate: bool = True):
        super().__init__()
        mid_dim: int = in_dim // projectionFactor
        self.unpool = nn.MaxUnpool2d(2)
        self.att = AttentionGate(ch_x=skip_ch, ch_g=in_dim) if (att_gate and skip_ch is not None) else None
        concat_ch = in_dim + (skip_ch if skip_ch is not None else 0)

        self.block0 = conv_block(concat_ch, mid_dim, kernel_size=3, padding=1)
        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)
        self.do = nn.Dropout(p=0.01)
        self.PReLU = nn.PReLU()

    def forward(self, args) -> Tensor:
        in_, indices, skip = args
        up = self.unpool(in_, indices)
        if self.att is not None:
            skip = self.att(skip, up)
        b0 = self.block0(torch.cat((up, skip), dim=1))
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)
        output = self.PReLU(up + do)
        return output


# ---------------------- ENet_enhance ----------------------

class ENet_enhance(nn.Module):
    """
    Enhanced ENet: SE + Lightweight Attention Gating + Dual-Head Supervision + Boundary Refinement
    API aligned with original ENet: ENet_enhance(input_dim, output_dim, number_of_convolutional_kernels,
    scaling_factor, use_SE=True, return_auxiliary_output=True)
    """
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        Fp: int = kwargs.get("factor", 4)
        K:  int = kwargs.get("kernels", 16)
        use_se: bool = kwargs.get("use_se", True)
        self.return_aux: bool = kwargs.get("return_aux", True)

        # Initial
        initial_channels: int = K - in_dim
        if initial_channels <= 0:
            raise ValueError(f"Number of kernels ({K}) must exceed input channels ({in_dim}).")
        self.conv0 = nn.Conv2d(in_dim, initial_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool0 = nn.MaxPool2d(2, return_indices=False, ceil_mode=False)

        # Down
        self.bottleneck1_0 = BottleNeckDownSampling(K, K * 4, Fp)
        self.bottleneck1_1 = nn.Sequential(
            BottleNeck(K * 4, K * 4, Fp, use_se=use_se),
            BottleNeck(K * 4, K * 4, Fp, use_se=use_se),
            BottleNeck(K * 4, K * 4, Fp, use_se=use_se),
            BottleNeck(K * 4, K * 4, Fp, use_se=use_se),
        )
        self.bottleneck2_0 = BottleNeckDownSampling(K * 4, K * 8, Fp)
        self.bottleneck2_1 = nn.Sequential(
            BottleNeck(K * 8, K * 8, Fp, dropoutRate=0.1, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dilation=2, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dropoutRate=0.1, asym=True, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dilation=4, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dropoutRate=0.1, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dilation=8, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dropoutRate=0.1, asym=True, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dilation=16, use_se=use_se),
        )

        # Middle
        self.bottleneck3 = nn.Sequential(
            BottleNeck(K * 8, K * 8, Fp, dropoutRate=0.1, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dilation=2, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dropoutRate=0.1, asym=True, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dilation=4, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dropoutRate=0.1, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dilation=8, use_se=use_se),
            BottleNeck(K * 8, K * 8, Fp, dropoutRate=0.1, asym=True, use_se=use_se),
            BottleNeck(K * 8, K * 4, Fp, dilation=16, dilate_last=True, use_se=use_se),
        )

        # Up (skip with AttentionGate)
        # bn3_out has K*4 channels, so in_dim=K*4; skip is bn1_out (K*4)
        self.bottleneck4 = nn.Sequential(
            BottleNeckUpSampling(K * 4, K * 4, Fp, skip_ch=K * 4, att_gate=True),  # ✅
            BottleNeck(K * 4, K * 4, Fp, dropoutRate=0.1, use_se=use_se),
            BottleNeck(K * 4, K, Fp, dropoutRate=0.1, use_se=use_se),
        )
        # The channel of bn4_out is K, so in_dim=K; skip is outputInitial(K).
        self.bottleneck5 = nn.Sequential(
            BottleNeckUpSampling(K, K, Fp, skip_ch=K, att_gate=True),  # ✅
            BottleNeck(K, K, Fp, dropoutRate=0.1, use_se=use_se),
        )

        # Deep supervision heads
        self.ds_head4 = nn.Conv2d(K,     out_dim, kernel_size=1)  # after bottleneck4 (H/4)
        self.ds_head3 = nn.Conv2d(K * 4, out_dim, kernel_size=1)  # after bottleneck3 (H/8)

        # Final refine & head
        self.br = BoundaryRefine(K)
        self.final = nn.Sequential(
            conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
            conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
            nn.Conv2d(K, out_dim, kernel_size=1)
        )

        print(f"> Initialized ENet_enhance ({in_dim=}->{out_dim=}) with factor={Fp}, kernels={K}, return_aux={self.return_aux}")

    def forward(self, input, return_aux: bool = None):
        if return_aux is None:
            return_aux = self.return_aux

        # Initial
        conv_0 = self.conv0(input)
        maxpool_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)  # [B, K, H/2, W/2]

        # Down
        bn1_0, indices_1 = self.bottleneck1_0(outputInitial)      # -> [B, 4K, H/4, W/4]
        bn1_out = self.bottleneck1_1(bn1_0)                       # skip1
        bn2_0, indices_2 = self.bottleneck2_0(bn1_out)            # -> [B, 8K, H/8, W/8]
        bn2_out = self.bottleneck2_1(bn2_0)

        # Middle
        bn3_out = self.bottleneck3(bn2_out)                       # -> [B, 4K, H/8, W/8]

        # Up
        bn4_out = self.bottleneck4((bn3_out, indices_2, bn1_out))         # -> [B, K, H/4, W/4]
        bn5_out = self.bottleneck5((bn4_out, indices_1, outputInitial))   # -> [B, K, H/2, W/2]

        # DS logits(Upsampling to H, W))
        aux4 = F.interpolate(self.ds_head4(bn4_out), scale_factor=4, mode="bilinear", align_corners=False)
        aux3 = F.interpolate(self.ds_head3(bn3_out), scale_factor=8, mode="bilinear", align_corners=False)

        # Final
        interpolated = F.interpolate(bn5_out, mode='nearest', scale_factor=2)  # -> [B, K, H, W]
        refined = self.br(interpolated)
        logits = self.final(refined)                                           # -> [B, out_dim, H, W]

        if return_aux:
            return logits, aux4, aux3
        return logits

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init)
