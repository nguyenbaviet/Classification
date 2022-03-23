import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "KnowledgeDistillationLoss",
    "JSDCrossEntropyLoss",
    "LabelSmoothingCrossEntropyLoss",
    "entropy_loss",
    "focal_loss",
]


class KnowledgeDistillationLoss(nn.Module):
    """
    Reference: https://nervanasystems.github.io/distiller/knowledge_distillation.html.

    Args:
        temperature (float): Temperature value used when calculating soft targets and logits.
    """

    def __init__(self, temperature=4.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_logit, teacher_logit):
        student_prob = F.softmax(student_logit, dim=-1)
        teacher_prob = F.softmax(teacher_prob, dim=-1).log()
        loss = F.kl_div(teacher_prob, student_prob, reduction="batchmean")
        return loss


class JSDCrossEntropyLoss(nn.Module):
    """
    Jensen-Shannon divergence + Cross-entropy loss.
    """

    def __init__(self, num_splits=3, alpha=12, clean_target_loss=nn.CrossEntropyLoss()):
        super(JSDCrossEntropyLoss, self).__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        self.cross_entropy_loss = clean_target_loss

    def forward(self, logit, target):
        split_size = logit.shape[0] // self.num_splits
        assert split_size * self.num_splits == logit.shape[0]
        logits_split = torch.split(logit, split_size)
        # Cross-entropy is only computed on clean images
        loss = self.cross_entropy_loss(logits_split[0], target[:split_size])
        probs = [F.softmax(logits, dim=1) for logits in logits_split]
        logp_mixture = torch.clamp(torch.stack(probs).mean(0), 1e-7, 1.0).log()
        loss += (
            self.alpha
            * sum(
                [
                    F.kl_div(logp_mixture, p_split, reduction="batchmean")
                    for p_split in probs
                ]
            )
            / len(probs)
        )
        return loss


def focal_loss(
    inputs,
    targets,
    alpha: float = -1,
    gamma: float = 2,
    act: str = "sigmoid",
    reduction: str = "none",
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    if act == "sigmoid":
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    elif act == "softmax":
        p = torch.softmax(inputs, 1)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        targets = torch.zeros(p.size(), device=p.device).scatter(
            1, targets.view(-1, 1), 1
        )
        ce_loss = ce_loss.view(-1, 1)
    p_t = p * targets + (1 - p) * (1 - targets)
    modulate = (1 - p_t) ** gamma
    loss = modulate * ce_loss

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Negative log-likelihood loss with label smoothing.

    Args:
        smoothing (float): label smoothing factor (default=0.1).
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logit, target):
        logprobs = F.log_softmax(logit, dim=-1)  # (bs, #classes)
        true_dist = torch.zeros_like(logit)
        true_dist.fill_(self.smoothing / (81313 - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))


class SeesawLoss(nn.Module):
    def __init__(self, class_num, p=0.8, q=2.0):
        super().__init__()
        N = torch.zeros(class_num)
        self.register_buffer("N", N)
        self.p = p
        self.q = q

    def forward(self, inputs, targets):
        """
        targets: index label
        """
        idx, counts = targets.unique(return_counts=True)
        self.N[idx] += counts
        Ni = self.N[targets]

        # mitigation factor
        M = self.N.unsqueeze(0) / Ni.unsqueeze(1)
        M[M >= 1] = 1
        M = M ** self.p

        # compensation factor
        inputs_i = inputs[:, targets]
        C = inputs / inputs_i
        C[C <= 1] = 1
        C = C ** self.q

        S = M * C

        loss = F.cross_entropy(inputs + torch.log(S), targets)
        return loss


@torch.jit.script
def entropy_loss(x, gamma: float = 1.0):
    entropy = -torch.mean(torch.sum(F.softmax(x, 1) * F.log_softmax(x, 1), 1), 0)
    return -gamma * entropy


def dice_loss(logit, target, act, eps=1e-6):
    if act == "sigmoid":
        prob = torch.sigmoid(logit)
    elif act == "softmax":
        prob = torch.softmax(logit, 1)
    intersection = (prob * target).sum((-1, -2)) + eps
    dice = (2.0 * intersection) / (prob.sum((-1, -2)) + target.sum((-1, -2)) + eps)
    return 1.0 - dice


def dice_metric(logit, target, act, threshold=0.5, eps=1e-6):
    if act == "sigmoid":
        if isinstance(threshold, (tuple, list)):
            pred = torch.stack(
                [
                    torch.sigmoid(logit[:, i, ...]) > th
                    for i, th in enumerate(threshold)
                ],
                1,
            )
            pred = pred.float()
        else:
            pred = (torch.sigmoid(logit) > threshold).float()

    intersection = (pred * target).sum((-1, -2)) + eps
    dice = (2.0 * intersection) / (pred.sum((-1, -2)) + target.sum((-1, -2)) + eps)
    return dice
