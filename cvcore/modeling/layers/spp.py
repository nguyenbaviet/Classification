import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SPPool2d"]


class SPPool2d(nn.Module):
    "Spatial Pyramid Pooling"

    def __init__(self, in_channels, pool_type="avg", pool_levels=(1, 2, 4)):
        super(SPPool2d, self).__init__()

        assert pool_type in ("avg", "max"), "Unsupported pooling type"
        self.pool_func = (
            F.adaptive_avg_pool2d if pool_type == "avg" else F.adaptive_max_pool2d
        )

        out_channels = 0
        for pool_level in pool_levels:
            out_channels += int(in_channels * math.pow(pool_level, 2))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_levels = pool_levels

    def forward(self, inputs):
        outputs = []
        for pool_level in self.pool_levels:
            output = self.pool_func(inputs, pool_level)
            output = torch.flatten(output, 1, -1)
            outputs.append(output)
        return torch.cat(outputs, 1)
