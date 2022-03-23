from typing import Union
from random import random

import numpy as np
import torch
from torch import nn
import albumentations as A

def albumentations_list(MAGN: int = 4):
    """
    Returns standard list of albumentations transforms, each of mangitude `MAGN`.
    
    Args:
        MAGN (int): Magnitude of each transform in the returned list.
    """
    M = MAGN
    transform_list = [
        # PIXEL-LEVEL
        A.RandomContrast(limit=M*.1, always_apply=True),
        A.RandomBrightness(limit=M*.1, always_apply=True), 
        A.Equalize(always_apply=True),
        A.OpticalDistortion(distort_limit=M*.2, shift_limit=M*.1, always_apply=True),
        A.RGBShift(r_shift_limit=M*10, g_shift_limit=M*10, b_shift_limit=M*10, always_apply=True),
        A.ISONoise(color_shift=(M*.01, M*.1),intensity=(M*.02, M*.2), always_apply=True),
        A.RandomFog(fog_coef_lower=M*.01, fog_coef_upper=M*.1, always_apply=True),
        A.CoarseDropout(max_holes=M*10, always_apply=True),
        A.GaussNoise(var_limit=(M,M*50), always_apply=True),

        # SPATIAL
        A.Rotate(always_apply=True),
        A.Transpose(always_apply=True),
        A.NoOp(always_apply=True),
        A.ElasticTransform(alpha=M*.25, sigma=M*3, alpha_affine=M*3, always_apply=True),
        A.GridDistortion(distort_limit=M*.075, always_apply=True)
    ]
    return transform_list
    
####################################### AlbumentationsRandAugment ####################################


class AlbumentationsRandAugment:
    """
    Item-wise RandAugment using Albumentations transforms. Use this to apply 
    RandAugment within `Dataset`.
    """
    def __init__(self,
                N_TFMS: int = 2, 
                MAGN: int = 4, 
                transform_list: list = None):
        """
        Args:
            N_TFMS (int): Number of transformation in each composition.
            MAGN (int): Magnitude of augmentation applied.
            tranform: List of K transformations to sample from.
        """
        
        if transform_list is None: transform_list = albumentations_list(MAGN)
            
        self.transform_list = transform_list
        self.MAGN = MAGN
        self.N_TFMS = N_TFMS
      
    
    def __call__(self, *args, force_apply: bool = False, **kwargs):
        """
        Returns a randomly sampled list of `N_TFMS` transforms from `transform_list`
        (default list provided if `None`).
        """
        sampled_tfms = np.random.choice(self.transform_list, self.N_TFMS)
        return list(sampled_tfms)
            

