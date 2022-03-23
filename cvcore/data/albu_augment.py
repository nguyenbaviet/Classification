import math
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from albumentations import DualTransform, ImageOnlyTransform, Normalize
from albumentations.augmentations import (
    random_crop,
    keypoint_scale,
    bbox_random_crop,
    keypoint_random_crop,
    resize,
)
from albumentations.augmentations.geometric.functional import (
    longest_max_size,
    smallest_max_size,
)
from albumentations.core.transforms_interface import BasicTransform

import numpy as np
from PIL import Image
import random
import torch
import torchvision


class ToTensorV2(BasicTransform):
    """Convert image and mask to `torch.Tensor`."""

    def __init__(self, always_apply=True, p=1.0):
        super(ToTensorV2, self).__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        return torch.from_numpy(img.transpose(2, 0, 1))

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return []

    def get_params_dependent_on_targets(self, params):
        return {}


class TenCropTTA(object):
    """
    Implement ImageNet's ten-crop test-time augmentation.
    """

    def __init__(
        self, scale_size=[256], img_size=(224, 224), interpolation=cv2.INTER_LINEAR
    ):
        self.scale_size = scale_size
        self.img_size = img_size
        self.interpolation = interpolation

        self.resize = ResizeShortestEdge(self.scale_size, self.interpolation)
        self.crop_func = torchvision.transforms.TenCrop(self.img_size)
        self.normalize = Normalize()
        self.totensor = ToTensorV2()

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Input image array.
        """
        img = np.asarray(img)
        results = []
        scaled_img = self.resize(image=img)["image"]
        crops = self.crop_func(Image.fromarray(scaled_img))
        for crop in crops:
            crop = np.asarray(crop)
            crop = self.normalize(image=crop)["image"]
            crop = self.totensor(image=crop)["image"]
            results.append(crop)
        return {"image": torch.stack(results, 0)}


class ResizeShortestEdge(ImageOnlyTransform):
    """
    Rescale an image so that shorest side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size (int): maximum size of smallest side of the image after the transformation.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self, max_size=[1024], interpolation=cv2.INTER_LINEAR, always_apply=False, p=1
    ):
        super(ResizeShortestEdge, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(self, img, **params):
        if len(self.max_size) > 1:
            max_size = np.random.randint(*self.max_size)
        else:
            max_size = self.max_size[0]
        return smallest_max_size(
            img, max_size=max_size, interpolation=self.interpolation
        )

    def get_transform_init_args_names(self):
        return ("max_size", "interpolation")


class ResizeLongestEdge(ImageOnlyTransform):
    """
    Rescale an image so that longest side is less than or equal to max_size,
    keeping the aspect ratio of the initial image.

    Args:
        max_size (int): maximum size of smallest side of the image after the transformation.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        func=max,
        max_size=1024,
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=1,
    ):
        super(ResizeLongestEdge, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size
        self.func = func

    def apply(self, img, **params):
        if self.func(img.shape[:2]) > self.max_size:
            img = longest_max_size(
                img,
                max_size=self.max_size,
                interpolation=self.interpolation,
            )
        return img

    def get_transform_init_args_names(self):
        return ("max_size", "interpolation")


class _BaseRandomSizedCrop(DualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    def __init__(self, height, width, interpolation, always_apply=False, p=1.0):
        super(_BaseRandomSizedCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(
        self,
        img,
        interpolation,
        crop_height=0,
        crop_width=0,
        h_start=0,
        w_start=0,
        **params
    ):
        crop = random_crop(img, crop_height, crop_width, h_start, w_start)
        if interpolation == "random":
            interpolation = np.random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        return resize(crop, self.height, self.width, interpolation)

    def apply_to_bbox(
        self,
        bbox,
        crop_height=0,
        crop_width=0,
        h_start=0,
        w_start=0,
        rows=0,
        cols=0,
        **params
    ):
        return bbox_random_crop(
            bbox, crop_height, crop_width, h_start, w_start, rows, cols
        )

    def apply_to_keypoint(
        self,
        keypoint,
        crop_height=0,
        crop_width=0,
        h_start=0,
        w_start=0,
        rows=0,
        cols=0,
        **params
    ):
        keypoint = keypoint_random_crop(
            keypoint, crop_height, crop_width, h_start, w_start, rows, cols
        )
        scale_x = self.width / crop_width
        scale_y = self.height / crop_height
        keypoint = keypoint_scale(keypoint, scale_x, scale_y)
        return keypoint


class RandomResizedCrop(_BaseRandomSizedCrop):
    """Torchvision's variant of crop a random part of the input and rescale it to some size.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        scale ((float, float)): range of size of the origin size cropped
        ratio ((float, float)): range of aspect ratio of the origin aspect ratio cropped
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        height,
        width,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=1.0,
    ):

        super(RandomResizedCrop, self).__init__(
            height=height,
            width=width,
            interpolation=interpolation,
            always_apply=always_apply,
            p=p,
        )
        self.scale = scale
        self.ratio = ratio

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        area = img.shape[0] * img.shape[1]

        for _attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))  # skipcq: PTC-W0028
            h = int(round(math.sqrt(target_area / aspect_ratio)))  # skipcq: PTC-W0028

            if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
                    "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
                }

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(self.ratio):
            w = img.shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img.shape[0]
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
        }

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "height", "width", "scale", "ratio", "interpolation"
