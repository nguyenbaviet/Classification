import math
import numpy as np
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from cvcore.utils import Registry
from cvcore.modeling.layers import GemPool2d
from cvcore.modeling.layers.ml_decoder import MLDecoder
from cvcore.utils.loss import FocalLoss, AsymmetricLossOptimized

OCCLUSION_CHECKERS = Registry("OCCLUSION_CHECKERS")

class fc_block(nn.Module):
    def __init__(self, inplanes, planes, drop_rate=0.15):
        super(fc_block, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.relu(x)
        return x

@OCCLUSION_CHECKERS.register()
class GEMPoolCNN(nn.Module):
    """
    Pre-trained ImageNet backbones with GeM pooling.
    """
    def __init__(self, cfg, num_classes):
        super(GEMPoolCNN, self).__init__()

        self.cfg = cfg.clone()

        pretrained = True if cfg.MODEL.BACKBONE.PRETRAINED == "imagenet" else False
        embeddings_dim = cfg.MODEL.BACKBONE.EMBEDDINGS_DIM
        self.backbone = create_model(
            cfg.MODEL.BACKBONE.ARCH,
            pretrained=pretrained,
            drop_path_rate=cfg.MODEL.BACKBONE.DROP_CONNECT,
        )
        self.backbone.reset_classifier(0, "")

        self.pool = nn.Sequential(
            GemPool2d(p=3.0),
            nn.Linear(self.backbone.num_features, embeddings_dim),
            nn.BatchNorm1d(embeddings_dim),
        )
        self.cls_head = nn.Linear(embeddings_dim, num_classes)
        self.loss_fn = FocalLoss()

    def _forward_features(self, images):
        features = self.backbone(images)
        features = self.pool(features)
        return features

    @autocast()
    def forward(self, images, labels=None):
        features = self._forward_features(images)
        logits = self.cls_head(features)
        # logits = logits.view(-1)
        if self.training:
            losses_dict = {}
            global_loss = self.loss_fn(logits, labels.float())
            losses_dict.update({"cls loss": global_loss})
            return losses_dict
        return logits

@OCCLUSION_CHECKERS.register()
class CustomEffNet(nn.Module):
    def __init__(self, cfg, num_classes):
        super(CustomEffNet, self).__init__()

        self.cfg = cfg.clone()

        pretrained = True if cfg.MODEL.BACKBONE.PRETRAINED == "imagenet" else False
        embeddings_dim = cfg.MODEL.BACKBONE.EMBEDDINGS_DIM
        self.backbone = create_model(
            cfg.MODEL.BACKBONE.ARCH,
            pretrained=pretrained,
            drop_path_rate=cfg.MODEL.BACKBONE.DROP_CONNECT,
        )
        in_features = self.backbone.get_classifier().in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        self.loss_fn = FocalLoss()

    @autocast()
    def forward(self, images, labels=None):
        logits = self.backbone(images)
        # logits = logits.view(-1)
        if self.training:
            losses_dict = {}
            global_loss = self.loss_fn(logits, labels.float())
            losses_dict.update({"cls loss": global_loss})
            return losses_dict
        return logits

@OCCLUSION_CHECKERS.register()
class CustomEffNet2(nn.Module):
    def __init__(self, cfg, num_classes):
        super(CustomEffNet2, self).__init__()

        self.cfg = cfg.clone()
        self.num_classes = num_classes

        pretrained = True if cfg.MODEL.BACKBONE.PRETRAINED == "imagenet" else False
        embeddings_dim = cfg.MODEL.BACKBONE.EMBEDDINGS_DIM
        self.backbone = create_model(
            cfg.MODEL.BACKBONE.ARCH,
            pretrained=pretrained,
            drop_path_rate=cfg.MODEL.BACKBONE.DROP_CONNECT,
        )
        self.backbone.reset_classifier(0, "")
        self.pool = nn.Sequential(
            GemPool2d(p=3.0),
            nn.Linear(self.backbone.num_features, embeddings_dim),
            nn.BatchNorm1d(embeddings_dim),
        )
        # self.cls_head = MLDecoder(num_classes=num_classes, initial_num_features=self.backbone.num_features)
        for i in range(num_classes):
            setattr(self, 'cls_head' + str(i).zfill(2), nn.Sequential(fc_block(embeddings_dim, 256), nn.Linear(256, 1)))

        self.loss_fn = FocalLoss()

    def _forward_features(self, images):
        features = self.backbone(images)
        features = self.pool(features)
        return features

    @autocast()
    def forward(self, images, labels=None):
        # logits = self.backbone(images)
        features = self._forward_features(images)
        # logits = self.cls_head(features)

        logits = []
        for i in range(self.num_classes):
            cls_head = getattr(self, 'cls_head' + str(i).zfill(2))
            logit = cls_head(features)
            logits.append(logit)
        logits = torch.cat(logits, dim=1)
        # logits = logits.view(-1)
        if self.training:
            losses_dict = {}
            global_loss = self.loss_fn(logits, labels.float())
            losses_dict.update({"cls loss": global_loss})
            return losses_dict
        return logits

@OCCLUSION_CHECKERS.register()
class CustomResNet(nn.Module):
    def __init__(self, cfg, num_classes):
        super(CustomResNet, self).__init__()

        self.cfg = cfg.clone()

        pretrained = True if cfg.MODEL.BACKBONE.PRETRAINED == "imagenet" else False
        embeddings_dim = cfg.MODEL.BACKBONE.EMBEDDINGS_DIM
        self.backbone = create_model(
            cfg.MODEL.BACKBONE.ARCH,
            pretrained=pretrained,
            drop_path_rate=cfg.MODEL.BACKBONE.DROP_CONNECT,
        )
        in_features = self.backbone.get_classifier().in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        self.loss_fn = FocalLoss()
        # self.loss_fn = AsymmetricLossOptimized()

    @autocast()
    def forward(self, images, labels=None):
        logits = self.backbone(images)
        # logits = logits.view(-1)
        if self.training:
            losses_dict = {}
            global_loss = self.loss_fn(logits, labels.float())
            losses_dict.update({"cls loss": global_loss})
            return losses_dict
        return logits

def build_cls_model(cfg, n_classes):
    return OCCLUSION_CHECKERS.get(cfg.MODEL.NAME)(cfg, n_classes)
