import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LocalBlock"]


class LocalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalBlock, self).__init__()
        self.in_channels = in_channels

        self.conv_enc = nn.Conv2d(in_channels, 128, 1, bias=True)
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, 1, bias=True),
            nn.Softplus(),
        )
        self.conv_dec = nn.Sequential(
            nn.Conv2d(128, in_channels, 1, bias=True), nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, inputs):
        dim_reduced_features = self.conv_enc(inputs)
        attention_scores = self.attention_net(inputs)

        if self.training:
            dim_expanded_features = self.conv_dec(dim_reduced_features)
            attn_prelogits = (
                attention_scores * F.normalize(dim_expanded_features, 1)
            ).mean((2, 3))
            attn_logits = self.fc(attn_prelogits)
            return dim_expanded_features, attn_logits

        return dim_reduced_features, attention_scores
