"""
RandAugment, AugMix
"""
from functools import partial
import random

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from albumentations.augmentations import functional
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
import numpy as np
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import torch


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def shear_x(img, v):
    assert -0.3 <= v <= 0.3
    height, width = img.shape[:2]

    matrix = np.float32([[1, v, 0], [0, 1, 0]])
    warp_fn = functional._maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height)
    )
    return warp_fn(img)


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def shear_y(img, v):
    assert -0.3 <= v <= 0.3
    height, width = img.shape[:2]

    matrix = np.float32([[1, 0, 0], [v, 1, 0]])
    warp_fn = functional._maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height)
    )
    return warp_fn(img)


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def translate_x(img, v):
    assert -0.45 <= v <= 0.45
    height, width = img.shape[:2]
    v = v * width

    matrix = np.float32([[1, 0, v], [0, 1, 0]])
    warp_fn = functional._maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height)
    )
    return warp_fn(img)


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def translate_y(img, v):
    assert -0.45 <= v <= 0.45
    height, width = img.shape[:2]
    v = v * height

    matrix = np.float32([[1, 0, 0], [0, 1, v]])
    warp_fn = functional._maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height)
    )
    return warp_fn(img)


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def rotate(img, v):
    assert -30 <= v <= 30
    height, width = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), v, 1.0)

    warp_fn = functional._maybe_process_in_chunks(
        cv2.warpAffine,
        M=matrix,
        dsize=(width, height),
        # flags=interpolation, borderMode=border_mode, borderValue=value
    )
    return warp_fn(img)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def invert(img, _):
    return functional.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def equalize(img, _):
    return functional.equalize(img, mode="pil")


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def flip(img, _):
    return functional.hflip(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    # assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def contrast(img, v):
    img = PIL.Image.fromarray(img)
    return np.asarray(Contrast(img, v))


def Color(img, v):  # [0.1,1.9]
    # assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def color(img, v):
    img = PIL.Image.fromarray(img)
    return np.asarray(Color(img, v))


def Brightness(img, v):  # [0.1,1.9]
    # assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def brightness(img, v):
    img = PIL.Image.fromarray(img)
    return np.asarray(Brightness(img, v))


def Sharpness(img, v):  # [0.1,1.9]
    # assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def sharpness(img, v):
    img = PIL.Image.fromarray(img)
    return np.asarray(Sharpness(img, v))


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0.0 <= v <= 0.2
    if v <= 0.0:
        return img
    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)
    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)
    xy = (x0, y0, x1, y1)
    # fill zero for mask
    if img.mode == "RGB":
        # color = (125, 123, 114)
        color = (0, 0, 0)
    else:
        color = (0,)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def cutout(img, v, fill_value=0):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    height, width = img.shape[:2]
    v = v * width

    x0 = np.random.uniform(width)
    y0 = np.random.uniform(height)
    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(width, x0 + v)
    y1 = min(height, y0 + v)
    xy = (x0, y0, x1, y1)
    # fill zero for mask
    img = img.copy()
    img[int(y0) : int(y1), int(x0) : int(x1), :] = 0
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def identity(img, v):
    return img


def make_augment_list(spatial_only=True):  # 16 operations and their ranges
    """
    Return a list of transformations used in RandAugment.

    Table 12 from FixMatch: Simplifying Semi-Supervised Learning with
    Consistency and Confidence (https://arxiv.org/pdf/2001.07685.pdf)
    """
    transforms_list = [
        # Spatial transforms
        # (Identity, 0.0, 1.0),
        (ShearX, 0.0, 0.3),  # 0
        (ShearY, 0.0, 0.3),  # 1
        (TranslateX, 0.0, 0.3),  # 2
        (TranslateY, 0.0, 0.3),  # 3
        (Rotate, 0, 30),  # 4
        (Cutout, 0, 0.5),  # 14
        (Flip, 0, 1),  # 15
    ]
    if spatial_only:
        return transforms_list
    transforms_list.extend(
        [
            # Colour transforms
            (AutoContrast, 0, 1),  # 5
            (Invert, 0, 1),  # 6
            (Equalize, 0, 1),  # 7
            (Solarize, 0, 255),  # 8
            (Posterize, 4, 8),  # 9
            (Contrast, 0.05, 0.95),  # 10
            (Color, 0.05, 0.95),  # 11
            (Brightness, 0.05, 0.95),  # 12
            (Sharpness, 0.05, 0.95),  # 13
            (SamplePairing(imgs), 0, 0.4),  # 16
        ]
    )
    return transforms_list


class RandAugment(object):
    """
    Generate a set of distortions.

    Args:
        n (int): Number of augmentation transformations to apply sequentially.
        m (int or List[int]): Magnitude for all the transformations.
    """

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = make_augment_list()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + f"(n={self.n}, m={self.m}, ops=[\n"
        for aug, minval, maxval in self.augment_list[:-1]:
            tmpstr += f"\t{aug.__name__}{minval, maxval},\n"
        aug, minval, maxval = self.augment_list[-1]
        tmpstr += f"\t{aug.__name__}{minval, maxval}])"
        return tmpstr

    def __call__(self, img):
        ops = random.sample(self.augment_list, k=self.n)

        if isinstance(self.m, list):
            magnitude = np.random.randint(*self.m)
        else:
            magnitude = self.m

        for op, minval, maxval in ops:
            val = (float(magnitude) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img


class RandAugmentAlbu(DualTransform):
    """
    Albumentations RandAugment.
    """

    def __init__(
        self,
        n=2,
        m=11,
        spatial_only=True,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=1.0,
    ):
        super(RandAugmentAlbu, self).__init__(always_apply, p)
        self.n = n
        self.m = m
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.transforms_list = [
            (shear_x, 0.0, 0.3),
            (shear_y, 0.0, 0.3),
            (translate_x, 0.0, 0.3),
            (translate_y, 0.0, 0.3),
            (rotate, 0, 30),
            (flip, 0, 1),
        ]
        if not spatial_only:
            self.transforms_list.extend(
                [
                    (brightness, 0.05, 0.95),
                    (contrast, 0.05, 0.95),
                    (color, 0.05, 0.95),
                    (sharpness, 0.05, 0.95),
                    (invert, 0, 1),
                    (equalize, 0, 1),
                    (functional.solarize, 0, 128),
                    # (functional.posterize, 0, 8),
                ]
            )

    def apply(self, img, vals, ops, **params):
        for op, val in zip(ops, vals):
            # print(op, val)
            img = op(img, val)
        return img

    def apply_to_mask(self, img, vals, ops, **params):
        for op, val in zip(ops, vals):
            op_name = op.__name__
            if op_name in ("invert", "equalize", "solarize", "posterize"):
                continue
            img = op(img, val)
        return img

    def get_params(self):
        ops = random.sample(self.transforms_list, self.n)

        if isinstance(self.m, (list, tuple)):
            magnitude = np.random.randint(*self.m)
        else:
            magnitude = self.m

        selected_ops = []
        vals = []
        for op, minval, maxval in ops:
            val = (float(magnitude) / 30) * float(maxval - minval) + minval
            op_name = op.__name__
            if op_name in (
                "shear_x",
                "shear_y",
                "translate_x",
                "translate_y",
                "rotate",
                "flip",
            ):
                if random.random() > 0.5:
                    val = -val
            selected_ops.append(op)
            vals.append(val)

        return {"vals": vals, "ops": selected_ops}

    def get_transform_init_args(self):
        return {
            "n": self.n,
            "m": self.m,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
            "value": self.value,
            "mask_value": self.mask_value,
        }


class Compose(object):
    """
    Sequentially apply a set of operations.
    """

    def __init__(self, m, ops):
        self.m = m
        self.ops = ops

    def __call__(self, img):
        for op, minval, maxval in self.ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img


class AugmentAndMix(object):
    def __init__(self, k=3, m=11, alpha=1.0, beta=1.0):
        self.k = k
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.augment_list = make_augment_list()

    def _transform(self, img):
        mixing_weights = np.random.dirichlet([self.alpha] * self.k)

        aug_img = np.zeros(img.size)
        for i, w in enumerate(mixing_weights):
            ops = random.sample(self.augment_list, self.k)
            op1 = Compose(self.m, ops[:1])
            op12 = Compose(self.m, ops[:-1])
            op123 = Compose(self.m, ops[::-1])
            chain = random.sample([op1, op12, op123], k=1)[0]
            aug_img += np.asarray(chain(img)) * w
        aug_img = PIL.Image.fromarray(aug_img.astype(np.uint8))

        augmix_img = PIL.Image.blend(aug_img, img, np.random.beta(self.beta, self.beta))
        return augmix_img

    def __call__(self, img):
        return (img, self._transform(img), self._transform(img))
