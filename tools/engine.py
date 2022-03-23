from PIL import Image
import math
import numpy as np
import os
import pandas as pd
import re
import sklearn
from sklearn import metrics
import json
import sys
import time
import torch

import torch.nn.functional as F
from torch.cuda.amp import autocast
from timm.utils import dispatch_clip_grad

from . import utils
from .metrics import F1_score, TPR_FPR
from face_dataset import TARGET

def freeze_model(cfg, model):
    """
    Freeze some or all parts of the model.
    """
    frozen_layers = cfg.MODEL.FREEZE_AT
    if len(frozen_layers) > 0:
        for name, parameter in model.named_parameters():
            if any([name.startswith(layer) for layer in frozen_layers]):
                parameter.requires_grad_(False)


def train_one_epoch(
    model,
    # loss_fn,
    optimizer,
    scheduler,
    data_loader,
    device,
    epoch,
    print_freq,
    gd_steps=1,
    scaler=None,
    clip_grad=False,
    model_ema=None,
    wandb_run=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    for i, (images, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # # Debug dataloader
        # images.mul_(data_loader.std).add_(data_loader.mean)
        # for i, img in enumerate(images):
        #     img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        #     pil_img = Image.fromarray(img)
        #     pil_img.save(f"img{i}.jpg")
        # import pdb
        # pdb.set_trace()

        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # with autocast():
        #     logits = model(images)

        # loss_dict = {}
        # global_loss = loss_fn(logits, targets.float())
        # loss_dict.update({"cls loss": global_loss})
        # losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # gradient accumulation
        losses = losses / gd_steps
        scaler.scale(losses).backward()
        if (i + 1) % gd_steps == 0:
            if clip_grad:
                scaler.unscale_(optimizer)
                dispatch_clip_grad(model.parameters(), value=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            if model_ema is not None:
                model_ema.update(model)

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if wandb_run is not None:
            wandb_run.log(
                {f"train/{k}": v
                 for k, v in loss_dict_reduced.items()})

    return metric_logger


@torch.no_grad()
def evaluate_classifier(model, valid_loader, wandb_run=None, eval_only=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Validate:"

    valid_probs = []
    valid_targets = []

    for images, targets in metric_logger.log_every(valid_loader, 100, header):
        torch.cuda.synchronize()

        model_time = time.time()
        with autocast():
            preds = model(images)
            pred_probs = torch.sigmoid(preds)  # live probability
            valid_probs.append(pred_probs.cpu())
            valid_targets.append(targets.cpu())
        model_time = time.time() - model_time

    valid_probs = torch.cat(valid_probs).numpy()
    valid_targets = torch.cat(valid_targets).numpy()
    threshold = 0.4
    y_pred = np.array(valid_probs >= threshold, dtype=float)
    if eval_only:
        org_dir = "/home/huyphan1/viet/liveness"
        df_label = pd.read_csv(os.path.join(org_dir,'test.csv'))
        df_res = pd.DataFrame(y_pred, columns = TARGET).astype(float)
        df_label = pd.concat([df_label.reset_index(drop=True),df_res.reset_index(drop=True)], axis=1)
        df_label.to_csv(os.path.join(org_dir,'eval_res.csv'), index = False)

    metrics = F1_score(y_pred, valid_targets)
    print(metrics)
    return metrics[0]


@torch.no_grad()
def update_soft_labels(model, valid_loader, wandb_run=None):
    print("Updating soft labels:")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Get probs:"

    valid_probs = []
    valid_targets = []

    for images, targets in metric_logger.log_every(valid_loader, 100, header):
        torch.cuda.synchronize()

        model_time = time.time()
        with autocast():
            preds = model(images)
            pred_probs = torch.sigmoid(preds)  # live probability
            valid_probs.append(pred_probs.cpu())
            valid_targets.append(targets.cpu())
        model_time = time.time() - model_time

    valid_probs = torch.cat(valid_probs).numpy()
    valid_targets = torch.cat(valid_targets).numpy()

    gamma = 0.1
    softened_labels = gamma*valid_probs + (1 - gamma)*valid_targets

    # softened_labels = valid_targets
    # softened_labels[np.where((valid_targets==1) & (valid_probs<0.3))] = 0.7

    return softened_labels