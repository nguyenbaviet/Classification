from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_pool import FastGlobalAvgPool2d, fast_global_avg_pool_2d

__all__ = ["OSMEBlock", "CrossXFPN", "c3s_regularizer"]


class OSMEBlock(nn.Module):
    """
    One-squeeze multiple-excites block.
    """

    def __init__(self, in_channels, reduction=16, num_excitations=2, fuse_type="cat"):
        super(OSMEBlock, self).__init__()
        reduction_channels = in_channels // reduction
        self.avg_pool = FastGlobalAvgPool2d(flatten=False)
        excitation_module = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Conv2d(in_channels, reduction_channels, 1, bias=False)),
                    ("bn1", nn.BatchNorm2d(reduction_channels)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("fc2", nn.Conv2d(reduction_channels, in_channels, 1, bias=False)),
                    ("bn2", nn.BatchNorm2d(in_channels)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )
        self.excites = nn.ModuleList(
            [excitation_module for i in range(num_excitations)]
        )

        assert fuse_type in {"avg", "cat", "sum"}
        self._fuse_type = fuse_type

    def forward(self, x):
        outputs = []
        x_squeeze = self.avg_pool(x)
        for excite in self.excites:
            x_excite = excite(x_squeeze)
            outputs.append(fast_global_avg_pool_2d(x_excite * x))
        if self._fuse_type == "avg":
            outputs = torch.stack(outputs).mean(0)
        elif self._fuse_type == "cat":
            outputs = torch.cat(outputs, 1)
        elif self._fuse_type == "sum":
            outputs = torch.stack(outputs).sum(0)
        return outputs


class CrossXFPN(nn.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, in_channels, out_channels):
        super(CrossXFPN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Arguments:
            features (list[Tensor]): feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            merged_features (Tensor): FPN feature map tensor.
        """
        assert len(x) == 2
        top_features = self.conv1(x[-1])
        merged_features = x[0] + F.interpolate(
            top_features, scale_factor=2, mode="nearest"
        )
        merged_features = self.conv2(merged_features)
        merged_features = self.bn(merged_features)
        return merged_features


def c3s_regularizer(features, gamma=0.5):
    """
    Loss to maximize the correlation of features from the
    same excitation module while minimizing the correlation of
    features from different excitation modules.
    """
    features = [fast_global_avg_pool_2d(feature) for feature in features]

    corr_matrix = torch.zeros(len(features), len(features)).to(features[0].device)
    for i in range(len(features)):
        for j in range(len(features)):
            corr_i_j = torch.mm(features[i], features[j].t()).mean()
            if i == j:
                corr_i_j *= -1.0
            corr_matrix[i, j] = corr_i_j

    reg = torch.mul(torch.sum(torch.triu(corr_matrix)), gamma)
    return reg
