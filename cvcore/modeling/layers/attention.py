import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm.models.layers import EcaModule
from timm.models.layers import create_act_layer


class LCAModule(nn.Module):
    """
    Local Channel Attention.
    """

    def __init__(self, kernel_size=3):
        super(LCAModule, self).__init__()

        padding = (kernel_size - 1) // 2
        # self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(1),
        )
        self.act = create_act_layer("sigmoid")

    def forward(self, x):
        x = x.mean((2, 3)).view(x.shape[0], 1, -1)
        x = self.conv(x)
        x = self.act(x)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        return x


class LSAModule(nn.Module):
    """
    Local Spatial Attention.
    """

    def __init__(self, channels, rd_ratio=8):
        super(LSAModule, self).__init__()

        rd_channels = channels // rd_ratio

        # self.conv1 = nn.Conv2d(channels, rd_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, rd_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rd_channels),
            nn.ReLU(),
        )
        # self.conv2 = nn.Conv2d(rd_channels, rd_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(rd_channels, rd_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(rd_channels),
            nn.ReLU(),
        )
        # self.conv3 = nn.Conv2d(
        #     rd_channels, rd_channels, kernel_size=3, padding=2, dilation=2
        # )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                rd_channels,
                rd_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(rd_channels),
            nn.ReLU(),
        )
        # self.conv4 = nn.Conv2d(
        #     rd_channels, rd_channels, kernel_size=3, padding=3, dilation=3
        # )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                rd_channels,
                rd_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                bias=False,
            ),
            nn.BatchNorm2d(rd_channels),
            nn.ReLU(),
        )
        # self.conv5 = nn.Conv2d(rd_channels * 4, 1, kernel_size=1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(rd_channels * 4, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        merged = torch.cat([x, self.conv2(x), self.conv3(x), self.conv4(x)], 1)
        merged = self.conv5(merged)
        return merged


class GCAModule(nn.Module):
    """
    Global Channel Attention.
    """

    def __init__(self, kernel_size=3):
        super(GCAModule, self).__init__()

        padding = (kernel_size - 1) // 2
        # self.conv_query = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding)
        self.conv_query = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )
        # self.conv_key = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding)
        self.conv_key = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        value = torch.flatten(x, start_dim=2)
        x = x.mean((2, 3)).view(x.shape[0], 1, -1)
        query = self.conv_query(x)
        key = self.conv_key(x)

        attention = torch.softmax(query.transpose(-2, -1) @ key, dim=-1)
        output = attention @ value
        output = output.reshape(N, C, H, W)
        # print(attention.shape)
        return output


class GSAModule(nn.Module):
    """
    Global Spatial Attention.
    """

    def __init__(self, channels, rd_ratio=8):
        super().__init__()

        rd_channels = channels // rd_ratio
        # self.conv_query = self.conv_key = self.conv_value = nn.Conv2d(
        #     channels, rd_channels, kernel_size=1
        # )
        self.conv_query = self.conv_key = self.conv_value = nn.Sequential(
            nn.Conv2d(channels, rd_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(rd_channels),
            nn.ReLU(),
        )
        # self.conv_out = nn.Conv2d(rd_channels, channels, kernel_size=1)
        self.conv_out = nn.Sequential(
            nn.Conv2d(rd_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        query = self.conv_query(x)
        N, C, H, W = query.shape
        query = torch.flatten(query, start_dim=2)
        key = self.conv_key(x)
        key = torch.flatten(key, start_dim=2)
        value = self.conv_value(x)
        value = torch.flatten(value, start_dim=2).permute(0, 2, 1)

        attention = torch.softmax(query.transpose(-2, -1) @ key, dim=-1)
        output = attention @ value
        output = output.permute(0, 2, 1).reshape(N, C, H, W)
        output = self.conv_out(output)
        # print(attention.shape)
        return output


class GLAModule(nn.Module):
    """
    Global-Local Attention.
    """

    def __init__(self, channels, kernel_size=3):
        super(GLAModule, self).__init__()

        self.lca_layer = LCAModule(kernel_size)
        self.lsa_layer = LSAModule(channels)
        self.gca_layer = GCAModule(kernel_size)
        self.gsa_layer = GSAModule(channels)
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # Local feature map
        local_feature = self.lca_layer(x) * x + x
        local_feature = self.lsa_layer(x) * local_feature + local_feature
        # Global feature map
        global_feature = self.gca_layer(x) * x
        global_feature = self.gsa_layer(x) * global_feature + global_feature

        feature = torch.stack([x, local_feature, global_feature], 0)
        weights = self.weights.view(len(feature), 1, 1, 1, 1)
        weights = torch.softmax(weights, 0)
        feature = (feature * weights).sum(0)
        return feature


if __name__ == "__main__":
    x = torch.randn(1, 2048, 7, 7)

    # layer = LCAModule()
    # layer = LSAModule(2048)
    # layer = GCAModule()
    # layer = GSAModule(2048)
    layer = GLAModule(2048)
    print(layer(x).shape)
