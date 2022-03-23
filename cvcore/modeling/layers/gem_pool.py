import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GemPool2d"]


class GemPool2d(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, trainable_p=False, clamp=True):
        super(GemPool2d, self).__init__()
        if trainable_p:
            self.p = nn.Parameter(torch.FloatTensor([p]))
        else:
            self.p = p
        self.eps = eps
        self.clamp = clamp

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "p=" + str(self.p)
        tmpstr += ", eps=" + str(self.eps)
        tmpstr += ")"
        return tmpstr

    def feat_mult(self):
        return 1

    def forward(self, x):
        if self.clamp:
            x = x.clamp(min=self.eps)
        x = x.pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)
        return x
