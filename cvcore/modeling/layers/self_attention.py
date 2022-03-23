import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SAModule"]


def kaiming_init(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)


def constant_init(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)


class SAModule(nn.Module):
    """
    Self-attention module.
    """

    def __init__(self, channels, reduction=1):
        super(SAModule, self).__init__()

        self.mid_channels = channels // reduction

        self.query = nn.Sequential(
            nn.Conv1d(channels, self.mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU(),
        )
        self.key = nn.Sequential(
            nn.Conv1d(channels, self.mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU(),
        )
        self.value = nn.Conv1d(
            channels,
            self.mid_channels,
            kernel_size=1,
            padding=0,
        )
        self.output = nn.Conv1d(
            self.mid_channels,
            channels,
            kernel_size=1,
            padding=0,
        )
        for conv in [self.query, self.key, self.value]:
            conv.apply(kaiming_init)
        self.output.apply(constant_init)

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)

        query_layer = self.query(x)
        key_layer = self.key(x)
        value_layer = self.value(x)
        attention_scores = torch.bmm(
            query_layer.permute(0, 2, 1).contiguous(), key_layer
        )
        attention_scores = attention_scores / math.sqrt(self.mid_channels)
        attention_scores = F.softmax(attention_scores, dim=1)
        context_layer = torch.bmm(value_layer, attention_scores)

        output_layer = self.output(context_layer)
        x = x + output_layer
        return x.view(*size).contiguous()
