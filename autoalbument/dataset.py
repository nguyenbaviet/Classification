import torch.utils.data
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

OCCLUSION_LABELS = ["label_mask","label_covered","label_hat","label_glass","label_sunglass","label_eyeclosed"]


class SearchDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        # Implement additional initialization logic if needed
        self.data_dir = "/home/huyphan1/quang/occlusion/faces"
        self.mode = "train"

        df = pd.read_csv(os.path.join(self.data_dir, f"{mode}.csv"))
        self.df = df

    def __len__(self):
        # Replace `...` with the actual implementation
        return len(self.df)

    def __getitem__(self, index):
        # Implement logic to get an image and its label using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `label` should be an integer in the range [0, model.num_classes - 1] where `model.num_classes`
        # is a value set in the `search.yaml` file.

        info = self.df.iloc[idx]
        img = cv2.imread(info["image"])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]
        # img = ToNumpy()(img)
        target = info[OCCLUSION_LABELS].to_numpy(dtype = np.float32)

        return img, target
