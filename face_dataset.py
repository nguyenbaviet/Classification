import json
import math
import os

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Resize, Normalize, ISONoise, MotionBlur
from cvcore.utils import Registry
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import Dataset
from timm.data import ToNumpy

LABELS = ["label_printed_color","label_photocopy","label_screen_capture", "label_corner_cut",  "label_normal"]
TARGET = ["target_printed_color", "target_photocopy","target_screen_capture", "target_corner_cut", "target_normal"]

class OcclusionDataset(Dataset):
    def __init__(self, cfg, mode="train"):

        super(OcclusionDataset, self).__init__()

        self.data_dir = cfg.DIRS.DATA
        self.mode = mode
        height = cfg.DATA.HEIGHT
        width = cfg.DATA.WIDTH
        interp = cfg.DATA.INTERP

        if self.mode == "train":
            self.aug = Compose([
                Resize(height, width, interpolation=interp),
                HorizontalFlip(p=0.5),
                ISONoise(color_shift=(0.15,0.35),
                        intensity=(0.2, 0.5), p=0.2),
                RandomBrightnessContrast(brightness_limit=0.2,
                        contrast_limit=0.2,
                        brightness_by_max=True,
                        always_apply=False, p=0.3),
                MotionBlur(blur_limit=5, p=0.2)
            ])
        elif self.mode == "valid" or self.mode == "test":
            self.aug = Resize(height, width, interpolation=interp)

        df = pd.read_csv(os.path.join(self.data_dir, f"{mode}.csv"))
        self.df = df

    # def get_class_weight(self):
    #     labels = self.df.replace({"target": CELEBA_SPOOF_LABELS})["target"]
    #     values, counts = np.unique(labels, return_counts=True)
    #     return counts[0]/counts[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        img = cv2.imread(info["image"])

        # # crop face
        # bbox = open(info["path"].replace(".jpg", "_BB.txt").replace(
        #     ".png", "_BB.txt")).read().split(" ")[:-1]
            
        # # x1 = min(int(bbox[0]), 0)
        # # y1 = min(int(bbox[1]), 0)
        # # x2 = min(int(bbox[2]), 0)
        # # y2 = min(int(bbox[3]), 0)
        # # h = y2 - y1
        # # w = x2 - x1
        # # if h > 10 and w > 10:
        # #     img = img[y1:y2, x1:x2, :]

        # if len(bbox)==4:
        #     real_h, real_w, _ = img.shape
        #     x1 = clamp(int(int(bbox[0])*(real_w / 224)), 0, real_w)
        #     y1 = clamp(int(int(bbox[1])*(real_h / 224)), 0, real_h)
        #     w1 = int(int(bbox[2])*(real_w / 224))
        #     h1 = int(int(bbox[3])*(real_h / 224))
        #     img = img[y1 : clamp(y1 + h1, 0, real_h), x1 : clamp(x1 + w1, 0, real_w), :]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.aug(image=img)["image"]
        img = ToNumpy()(img)
        target = info[LABELS].to_numpy(dtype = np.float32)
        return img, target
        # return torch.FloatTensor(img), torch.FloatTensor(target)

def clamp(x, min_x, max_x):
    return min(max(x, min_x), max_x)