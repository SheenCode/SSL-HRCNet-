import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwisePointwiseConv1D(nn.Module):
    """
    Depthwise separable 1D convolution:
    depthwise conv -> pointwise conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2

        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LearnableResidualConvNet(nn.Module):
    """
    ConvNet1 / ConvNet3:
    x'  = x + alpha1 * DConv(x)
    x'' = x' + alpha2 * PConv(x')
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2

        # DConv branch
        self.dconv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )

        # pointwise after residual 1
        self.pconv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        # projection for residual alignment
        if stride != 1 or in_channels != out_channels:
            self.proj_to_in = nn.Conv1d(
                in_channels, in_channels, kernel_size=1, stride=stride, bias=False
            )
            self.proj_to_out = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.proj_to_in = nn.Identity()
            self.proj_to_out = nn.Identity()

        # learnable scalar weights
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # first residual: x' = x + alpha1 * DConv(x)
        identity1 = self.proj_to_in(x)
        d = self.dconv(x)
        d = self.bn1(d)
        x1 = identity1 + self.alpha1 * d
        x1 = self.act(x1)

        # second residual: x'' = x' + alpha2 * PConv(x')
        identity2 = self.proj_to_out(x1)
        p = self.pconv(x1)
        p = self.bn2(p)
        out = identity2 + self.alpha2 * p
        out = self.act(out)

        return out


class DownsampleConvNet(nn.Module):
    """
    ConvNet2:
    temporal downsampling with stride=2, no residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        self.block = DepthwisePointwiseConv1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, x):
        return self.block(x)


class HierarchicalLearnableResidualBlock(nn.Module):
    """
    One hierarchical block:
    ConvNet1 (k=3, residual)
    ConvNet2 (k=5, stride=2, no residual)
    ConvNet3 (k=7, residual)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convnet1 = LearnableResidualConvNet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1
        )
        self.convnet2 = DownsampleConvNet(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=2
        )
        self.convnet3 = LearnableResidualConvNet(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)  # 对齐
    def forward(self, x):
        identity = self.shortcut(x)
        x = self.convnet1(x)
        x = self.convnet2(x)
        x = self.convnet3(x)
        return x + identity


class FeatureEncoder(nn.Module):
    """
    Initial block + 3 stacked hierarchical blocks
    Output channels: 128
    """
    def __init__(self, in_channels=1):
        super().__init__()

        self.initial_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.initial_bn = nn.BatchNorm1d(16)
        self.initial_act = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.stage1 = HierarchicalLearnableResidualBlock(16, 32)
        self.stage2 = HierarchicalLearnableResidualBlock(32, 64)
        self.stage3 = HierarchicalLearnableResidualBlock(64, 128)

    def forward(self, x):
        """
        x: [B, 1, L]
        returns: [B, 128, L']
        """
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_act(x)
        x = self.initial_pool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class CrossAttentionFusion(nn.Module):
    """
    Attention-based cross fusion module.
    Input:
        F_rri     : [B, C, L]
        F_rpeaks  : [B, C, L]
    Output:
        fused     : [B, 2C, L]
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

        self.q_rri = nn.Linear(embed_dim, embed_dim)
        self.k_rri = nn.Linear(embed_dim, embed_dim)
        self.v_rri = nn.Linear(embed_dim, embed_dim)

        self.q_rpeak = nn.Linear(embed_dim, embed_dim)
        self.k_rpeak = nn.Linear(embed_dim, embed_dim)
        self.v_rpeak = nn.Linear(embed_dim, embed_dim)

    def forward(self, f_rri, f_rpeak):
        # [B, C, L] -> [B, L, C]
        f_rri = f_rri.transpose(1, 2)
        f_rpeak = f_rpeak.transpose(1, 2)

        # projections
        q_rri = self.q_rri(f_rri)
        k_rri = self.k_rri(f_rri)
        v_rri = self.v_rri(f_rri)

        q_rpeak = self.q_rpeak(f_rpeak)
        k_rpeak = self.k_rpeak(f_rpeak)
        v_rpeak = self.v_rpeak(f_rpeak)

        # A_{R-peaks -> RRIs} = Softmax(Q_RRIs K_R-peaks^T / sqrt(dk))
        attn_rpeak_to_rri = torch.matmul(q_rri, k_rpeak.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn_rpeak_to_rri = F.softmax(attn_rpeak_to_rri, dim=-1)

        # A_{RRIs -> R-peaks} = Softmax(Q_R-peaks K_RRIs^T / sqrt(dk))
        attn_rri_to_rpeak = torch.matmul(q_rpeak, k_rri.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn_rri_to_rpeak = F.softmax(attn_rri_to_rpeak, dim=-1)

        f_rri_cross = torch.matmul(attn_rpeak_to_rri, v_rri)
        f_rpeak_cross = torch.matmul(attn_rri_to_rpeak, v_rpeak)

        fused = torch.cat([f_rri_cross, f_rpeak_cross], dim=-1)  # [B, L, 2C]
        fused = fused.transpose(1, 2)  # [B, 2C, L]
        return fused

