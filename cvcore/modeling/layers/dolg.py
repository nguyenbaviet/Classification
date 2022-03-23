import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels=1024, atrous_rates=[6, 12, 18], out_channels=512):
        super(ASPP, self).__init__()

        modules = []
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, dilation=rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SpatialAttention2d(nn.Module):
    def __init__(self, in_channels=1024):
        super(SpatialAttention2d, self).__init__()

        self.conv_enc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1, bias=True),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.conv_enc(x)
        attention_scores = self.attention(x)
        x = F.normalize(x) * attention_scores
        return x
