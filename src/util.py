import math

import torch
import torch.nn as nn

# Feature Rectify Module
from torch.nn.init import trunc_normal_


def printShape(t, tName):
    print(f"{tName}.shape->", t.shape)


class ChannelWeights(nn.Module):  # # 2 B C 1 1
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # bs 6 1 1
        avg = self.avg_pool(x).view(B, self.dim * 2)  # bs 2dim
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def forward(self, x):
        x1, x2 = x
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        # print(channel_weights.shape, spatial_weights.shape)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, u1, u2):
        B, N, C = u1.shape  # 1 14400 256

        q1 = u1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = u2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        k1, v1 = self.kv1(u1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(u2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()


        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)


        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        vv1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        vv2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return vv1, vv2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()

        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)

        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)

        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        # 分成两块 维度 - 1
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)  # torch.Size([1, 14400, 256])
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)  # torch.Size([1, 14400, 256])

        vv1, vv2 = self.cross_attn(u1, u2)

        y1 = torch.cat((y1, vv1), dim=-1)
        y2 = torch.cat((y2, vv2), dim=-1)

        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)

    def forward(self, x):
        x1, x2 = x
        B, C, H, W = x1.shape

        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)

        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)

        return merge


class MFFM(nn.Module):
    def __init__(self, embed_dims, num_heads):
        super().__init__()
        self.FRMs = FeatureRectifyModule(dim=embed_dims, reduction=1)
        self.FFMs = FeatureFusionModule(dim=embed_dims, reduction=1, num_heads=num_heads, norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        out_x1, out_x2 = self.FRMs(x)
        x_fused = self.FFMs([out_x1, out_x2])

        return [out_x1, out_x2], x_fused


if __name__ == '__main__':
    in_batch, inchannel, in_h, in_w = 1, 256, 120, 120

    x = torch.randn(in_batch, inchannel, in_h, in_w)
    y = torch.randn(in_batch, inchannel, in_h, in_w)

    FRM = FeatureRectifyModule(inchannel)
    out_x1, out_x2 = FRM([x, y])
    # print(out_x1.shape, out_x2.shape)  # torch.Size([1, 256, 120, 120]) torch.Size([1, 256, 120, 120])

    dim = inchannel
    head_num = 8
    FFM = FeatureFusionModule(dim=dim, reduction=1, num_heads=1, norm_layer=nn.BatchNorm2d)
    merge = FFM([out_x1, out_x2])  # torch.Size([1, 256, 120, 120])
    # print(merge.shape)

    '''
    mffm = MFFM(inchannel, 1)
    [x1, x2], merge = mffm([x, y])

    print(x1.shape, x2.shape,merge.shape)  # torch.Size([1, 256, 120, 120]) torch.Size([1, 256, 120, 120]) torch.Size([1, 256, 120, 120])
    print(mffm)
    '''
