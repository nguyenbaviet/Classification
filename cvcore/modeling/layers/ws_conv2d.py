import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WSConv2d"]


class WSConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(WSConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
        )
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

    @classmethod
    def convert_wsconv2d(cls, module):
        """
        Recursively traverse module and its children to replace all instances of
        ``Conv2d`` with `WSConv2d`.
        Args:
            module (nn.Module): input module
        """
        mod = module
        if isinstance(module, nn.Conv2d):
            mod = cls(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias,
            )
            mod.weight = module.weight
            mod.bias = module.bias
        else:
            for name, child in module.named_children():
                new_child = convert_wsconv2d(child)
                if new_child is not child:
                    mod.add_module(name, new_child)
        del module
        return mod
