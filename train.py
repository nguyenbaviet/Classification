import argparse
import datetime
import os
import time
import sys
from numpy import inf
import pandas as pd
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.cuda.amp import GradScaler
import torch.utils.data

from cvcore.config import get_cfg
from cvcore.solver import make_lr_scheduler, make_optimizer
from face_dataset import OcclusionDataset
from cvcore.utils import seed_all, worker_init_reset_seed
from cvcore.utils.loss import FocalLoss, AsymmetricLossOptimized
from models import build_cls_model
from tools.engine import train_one_epoch, evaluate_classifier, freeze_model, update_soft_labels
import tools.utils as utils
from face_dataset import LABELS

from timm.utils.model_ema import ModelEmaV2
from timm.data.loader import PrefetchLoader

def create_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config",
                        default="",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--start_epoch",
                        default=0,
                        type=int,
                        help="start epoch")
    parser.add_argument("--print-freq",
                        default=50,
                        type=int,
                        help="print frequency")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--finetune",
        "-ft",
        action="store_true",
        help=
        "whether to attempt to resume from the checkpoint optimizer, scheduler and epoch",
    )
    parser.add_argument(
        "--eval-only",
        help="Run model evaluation",
        action="store_true",
    )
    parser.add_argument(
        "--soft_labels",
        help="Use soft labels",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--test",
        help="Run model evaluation on test dataset",
        action="store_true",
    )
    parser.add_argument("--num-gpu",
                        type=int,
                        default=1,
                        help="Number of GPUS to use")
    parser.add_argument(
        "--clip-grad",
        dest="clip_grad",
        help="apply gradient clipping",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        default=False,
        help="log training and validation metrics to wandb",
    )
    parser.add_argument("--wandb-id", default="", help="wandb id for resuming")
    # distributed training parameters
    parser.add_argument("--world-size",
                        default=1,
                        type=int,
                        help="number of distributed processes")
    parser.add_argument("--dist-url",
                        default="env://",
                        help="url used to set up distributed training")

    args = parser.parse_args()
    return args


def main(args, cfg, wandb_run=None):
    imgs_per_gpu = cfg.TRAIN.BATCH_SIZE // args.world_size
    workers_per_gpu = cfg.SYSTEM.NUM_WORKERS // args.world_size
    kfold = cfg.DATA.KFOLD.ENABLED
    exp_name = cfg.EXPERIMENT
    if kfold:
        exp_name += f"_fold{cfg.DATA.FOLD}"
    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    dataset = OcclusionDataset(cfg, "train")
    # class_weights = dataset.get_class_weight()
    num_classes = cfg.MODEL.NUM_CLASSES
    if args.eval_only:
        print("Using test set")
        dataset_test = OcclusionDataset(cfg, "test")
    else:
        print("Using validation set")
        dataset_test = OcclusionDataset(cfg, "valid")

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler,
                                                        imgs_per_gpu,
                                                        drop_last=True)
    apply_cutout = cfg.DATA.CUTOUT
    if apply_cutout:
        print("Adding cutout to augmentation")
        re_prob = 0.5
        re_mode = "pixel"
        re_count = 3
    else:
        re_prob = 0.0
        re_mode = "const"
        re_count = 1
    data_loader = PrefetchLoader(
        torch.utils.data.DataLoader(
            dataset,
            batch_sampler=train_batch_sampler,
            num_workers=workers_per_gpu,
            worker_init_fn=worker_init_reset_seed,
        ),
        fp16=True,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
    )
    data_loader_test = PrefetchLoader(
        torch.utils.data.DataLoader(
            dataset_test,
            batch_size=imgs_per_gpu,
            sampler=test_sampler,
            num_workers=workers_per_gpu,
        ),
        fp16=True,
    )
    if args.soft_labels:
        print("Using soft labels")
        soft_label_sampler = torch.utils.data.SequentialSampler(dataset)
        data_loader_soft_label = PrefetchLoader(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=imgs_per_gpu,
                sampler=soft_label_sampler,
                num_workers=workers_per_gpu,
            ),
            fp16=True,
        )

    model = build_cls_model(cfg, num_classes)
    print(
        f"Created model {cfg.EXPERIMENT} - param count: {sum([m.numel() for m in model.parameters()])}"
    )
    freeze_model(cfg, model)
    model.to(device)
    model_ema = None
    use_ema = cfg.OPT.SWA.ENABLED
    if use_ema:
        print("Creating exponential moving average model")
        model_ema = ModelEmaV2(model, decay=cfg.OPT.SWA.DECAY_RATE)

    scaler = GradScaler()
    optimizer = make_optimizer(cfg, model)
    # loss_fn = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=1, clip=0)

    model_without_ddp = model
    if args.distributed:
        if not cfg.MODEL.FREEZE_BATCHNORM.ENABLED:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    lr_scheduler = make_lr_scheduler(cfg, optimizer, data_loader)

    # best_metric = -inf
    best_metric = inf

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        if model_ema:
            if "model_ema" not in checkpoint.keys():
                model_ema.module.load_state_dict(checkpoint["model"])
            else:
                model_ema.module.load_state_dict(checkpoint["model_ema"])
        if args.finetune:
            print("Skip loading optimizer and scheduler state dicts")
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            best_metric = checkpoint["best_metric"]
            print(f"Epoch: {args.start_epoch} - Best metric: {best_metric}")

    if args.eval_only:
        model_path = os.path.join(cfg.DIRS.WEIGHTS, f"best_{exp_name}.pth")
        checkpoint = torch.load(model_path, map_location='cuda:0')
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        if use_ema:
            model_without_ddp.load_state_dict(checkpoint["model_ema"])
        evaluate_classifier(model_without_ddp,
                            data_loader_test,
                            wandb_run=wandb_run,
                            eval_only=True)
        return
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, cfg.TRAIN.EPOCHS):

        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model,
            # loss_fn,
            optimizer,
            lr_scheduler,
            data_loader,
            device,
            epoch,
            args.print_freq,
            gd_steps=cfg.OPT.GD_STEPS,
            scaler=scaler,
            clip_grad=args.clip_grad,
            model_ema=model_ema,
            wandb_run=wandb_run,
        )
        # evaluate_classifier after every epoch
        if use_ema:
            epoch_metric = evaluate_classifier(
                model_without_ddp,
                data_loader_test,
                wandb_run=wandb_run,
            )
        else:
            epoch_metric = evaluate_classifier(
                model_without_ddp,
                data_loader_test,
                wandb_run=wandb_run,
            )

        #Update soft labels:
        if args.soft_labels and epoch > cfg.TRAIN.EPOCHS - 5 and epoch < cfg.TRAIN.EPOCHS - 1:
            softened_labels = update_soft_labels(
                model_without_ddp,
                data_loader_soft_label,
                wandb_run=wandb_run,
            )
            data_loader.loader.dataset.df[LABELS] = softened_labels
            data_loader_soft_label.loader.dataset.df[LABELS] = softened_labels

        # Checkpoint
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": args,
            "epoch": epoch,
            "best_metric": best_metric,
        }
        if use_ema:
            checkpoint.update({
                "model_ema": model_ema.module.state_dict(),
            })
        # if epoch_metric > best_metric:
        if epoch_metric < best_metric:
            best_metric = epoch_metric
            checkpoint.update({"best_metric": best_metric})
            utils.save_on_master(
                checkpoint,
                os.path.join(cfg.DIRS.WEIGHTS, f"best_{exp_name}.pth"),
            )
        utils.save_on_master(
            checkpoint,
            os.path.join(cfg.DIRS.WEIGHTS, exp_name + ".pth"),
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    sys.exit(0)


if __name__ == "__main__":
    args = create_args()
    # Initialize env.
    utils.init_distributed_mode(args)
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    seed_all()

    if args.log_wandb and args.rank == 0:
        import wandb

        if args.wandb_id != "":
            run = wandb.init(project="landmark", id=args.wandb_id, resume=True)
        else:
            run = wandb.init(project="landmark")
        run.config.update(
            {
                "warmup_epochs": cfg.OPT.WARMUP_EPOCHS,
                "epochs": cfg.TRAIN.EPOCHS,
                "batch_size": cfg.TRAIN.BATCH_SIZE,
                "backbone_lr": cfg.OPT.BACKBONE_LR,
                "meta_arch": cfg.MODEL.NAME,
                "backbone": cfg.MODEL.BACKBONE.ARCH,
            },
            allow_val_change=True,
        )
    else:
        run = None

    # Make dirs
    for _dir in ["WEIGHTS", "OUTPUTS", "LOGS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.makedirs(cfg.DIRS[_dir], exist_ok=True)

    main(args, cfg, run)
