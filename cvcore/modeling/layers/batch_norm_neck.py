import torch
import torch.nn as nn

__all__ = ["BNNeck"]


class BNNeck(nn.Module):
    """
    Implement batchnorm neck.
    """

    def __init__(self, in_channels, out_channels):
        super(BNNeck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = nn.BatchNorm1d(in_channels)
        self.fc = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        bnneck_features = self.bn(x)
        bnneck_logits = self.fc(bnneck_features)
        return bnneck_features, bnneck_logits
