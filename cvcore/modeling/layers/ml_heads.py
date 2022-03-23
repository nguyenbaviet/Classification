import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvcore.utils import Registry

__all__ = [
    "ArcFace",
    "CosFace",
    "build_ml_head",
    "weight_regularizer",
    "CurricularFace",
]

ML_HEAD_REGISTRY = Registry("ML_HEAD")
ML_HEAD_REGISTRY.__doc__ = """
Registry for metric learning heads.
"""


def build_ml_head(cfg, in_channels):
    head_name = cfg.MODEL.ML_HEAD.NAME
    out_channels = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.ML_HEAD.SCALER > 0:
        s = cfg.MODEL.ML_HEAD.SCALER
    else:
        s = math.sqrt(in_channels)
    m = cfg.MODEL.ML_HEAD.MARGIN
    num_centers = cfg.MODEL.ML_HEAD.NUM_CENTERS
    head = ML_HEAD_REGISTRY.get(head_name)(in_channels, out_channels, s, m, num_centers)
    return head


@ML_HEAD_REGISTRY.register()
class MagFace(nn.Module):
    """
    This module implements MagFace.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        s=32.0,
        m=0.5,
        num_centers=1,
        lamb_g=35,
        l_a=10,
        u_a=110,
        l_m=0.15,
        u_m=0.45,
    ):
        super(MagFace, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s
        self.lamb_g = lamb_g
        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m
        self.num_centers = num_centers
        self.easy_margin = False

        self.weight = nn.Parameter(
            torch.Tensor(out_channels * num_centers, in_channels)
        )
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_channels="
            + str(self.in_channels)
            + ", out_channels="
            + str(self.out_channels)
            + ", s="
            + str(self.s)
            + ", lamb_g="
            + str(self.lamb_g)
            + ", l_a="
            + str(self.l_a)
            + ", u_a="
            + str(self.u_a)
            + ", num_centers="
            + str(self.num_centers)
            + ")"
        )

    def forward(self, inputs, labels):
        # cos(theta)
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        cosine = cosine.float()
        if self.num_centers > 1:
            cosine = cosine.view(cosine.size(0), self.num_centers, self.out_channels)
            cosine = F.softmax(cosine * self.s, 1) * cosine
            cosine = cosine.sum(1)
        if not self.training or labels is None:
            return cosine * self.s, None
        # adaptive margins
        magnitude = inputs.norm(p=2, dim=1, keepdim=True).clamp(
            self.l_a, self.u_a
        )  # magnitude is bounded in [l_a, u_a]
        ada_margin = ((self.u_m - self.l_m) / (self.u_a - self.l_a)) * (
            magnitude - self.l_a
        ) + self.l_m
        cos_m = torch.cos(ada_margin)
        sin_m = torch.sin(ada_margin)
        th = torch.cos(math.pi - ada_margin)
        mm = torch.sin(math.pi - ada_margin) * ada_margin
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m - sine * sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - th) > 0, phi, cosine - mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        # magnitude regularizer
        magnitude_reg = (1.0 / magnitude) + (1.0 / math.pow(self.u_a, 2)) * magnitude
        magnitude_reg = magnitude_reg * self.lamb_g
        return output, magnitude_reg


@ML_HEAD_REGISTRY.register()
class ArcFace(nn.Module):
    """
    This module implements ArcFace.
    """

    def __init__(self, in_channels, out_channels, s=32.0, m=0.5, num_centers=1):
        super(ArcFace, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s
        self.m = m
        self.num_centers = num_centers

        self.weight = nn.Parameter(
            torch.Tensor(out_channels * num_centers, in_channels)
        )
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_channels="
            + str(self.in_channels)
            + ", out_channels="
            + str(self.out_channels)
            + ", s="
            + str(self.s)
            + ", m="
            + str(self.m)
            + ", num_centers="
            + str(self.num_centers)
            + ")"
        )

    def forward(self, inputs, labels):
        # cos(theta)
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        cosine = cosine.float()
        if self.num_centers > 1:
            cosine = cosine.view(cosine.size(0), self.num_centers, self.out_channels)
            cosine = F.softmax(cosine * self.s, 1) * cosine
            cosine = cosine.sum(1)
        if not self.training or labels is None:
            return cosine * self.s, None
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output, None


@ML_HEAD_REGISTRY.register()
class CosFace(nn.Module):
    """
    This module implements CosFace (https://arxiv.org/pdf/1801.09414.pdf).
    """

    def __init__(self, in_channels, out_channels, s=64.0, m=0.35, num_centers=1):
        super(CosFace, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s = s
        self.m = m
        self.num_centers = num_centers

        self.weight = nn.Parameter(
            torch.FloatTensor(out_channels * num_centers, in_channels)
        )
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_channels="
            + str(self.in_channels)
            + ", out_channels="
            + str(self.out_channels)
            + ", s="
            + str(self.s)
            + ", m="
            + str(self.m)
            + ")"
        )

    def forward(self, inputs, labels):
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        cosine = cosine.float()
        if self.num_centers > 1:
            cosine = cosine.view(cosine.size(0), self.num_centers, self.out_channels)
            cosine = F.softmax(cosine * self.s, 1) * cosine
            cosine = cosine.sum(1)
        if not self.training:
            return cosine * self.s, None
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output, None


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


@ML_HEAD_REGISTRY.register()
class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5, num_centers=1):
        super(CurricularFace, self).__init__()
        print("----------- init ", s, m)
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer("t", torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.float()
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        if not self.training or label is None:
            return cos_theta * self.s, None
        # with torch.no_grad():
        #     origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = (
            target_logit * self.cos_m - sin_theta * self.sin_m
        )  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(
            target_logit > self.threshold, cos_theta_m, target_logit - self.mm
        )

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        # return output, origin_cos * self.s
        return output, None


def logsumexp(x, keep_mask=None, add_one=True, dim=1):
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, float("-inf"))
    if add_one:
        zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(
            dim
        )
        x = torch.cat([x, zeros], dim=dim)

    output = torch.logsumexp(x, dim=dim, keepdim=True)
    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    return output


class MultiSimilarityLoss(nn.Module):
    """
    Modified from https://github.com/MalongTech/research-ms-loss/blob/master/ret_benchmark/losses/multi_similarity_loss.py
    """

    def __init__(self, eps=0.1, threshold=0.5, alpha=2, beta=50):
        super(MultiSimilarityLoss, self).__init__()

        self.eps = eps
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta

    def forward(self, features, labels):
        # obtain similarities matrix
        features = F.normalize(features)
        sim_mat = torch.matmul(features, features.T)
        pos_sim = sim_mat
        neg_sim = sim_mat.clone()

        # mine hard negative/ positive pairs
        matches = (labels.unsqueeze(1) == labels.unsqueeze(0)).byte()
        diffs = matches ^ 1
        a1, p = torch.where(matches)
        a2, n = torch.where(diffs)
        pos_sim[a2, n] = float("inf")
        neg_sim[a1, p] = float("-inf")
        neg_mask = neg_sim > torch.min(pos_sim - self.eps, 1, keepdim=True)[0]
        pos_mask = pos_sim < torch.max(neg_sim + self.eps, 1, keepdim=True)[0]

        pos_loss = (1.0 / self.alpha) * logsumexp(
            self.alpha * (self.threshold - pos_sim),
            keep_mask=pos_mask.bool(),
            add_one=True,
        )
        neg_loss = (1.0 / self.beta) * logsumexp(
            self.beta * (neg_sim - self.threshold),
            keep_mask=neg_mask.bool(),
            add_one=True,
        )
        loss = (pos_loss + neg_loss).mean()
        return loss


@torch.jit.script
def weight_regularizer(weight, num_centers: int = 1, tau: float = 0.2):
    """
    Loss to encourage similar centers to merge with each other.
    """
    weight = F.normalize(weight)
    weight = weight.view(
        num_centers, -1, weight.size(1)
    )  # shape(num_centers, num_classes, emb_dim)
    reg = weight.unsqueeze(1) - weight.unsqueeze(0)
    reg = (
        tau
        * torch.pow(reg, 2).sum()
        / (num_centers * (num_centers - 1) * weight.size(1))
    )
    return reg
