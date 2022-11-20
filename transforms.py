import numpy as np
import random
import torch
import torch.nn.functional as F
from torchaudio.transforms import FrequencyMasking, TimeMasking
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform


class FrequencyMaskingTensor(ImageOnlyTransform):
    def __init__(self, mask_max=72, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mask_max = mask_max
        self.mask = FrequencyMasking(self.mask_max)

    def apply(self, img: torch.Tensor, **params):
        return self.mask(img)

    def get_transform_init_args_names(self):
        return {'mask_max': self.mask_max}


class TimeMaskingTensor(ImageOnlyTransform):
    def __init__(self, mask_max=256, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mask_max = mask_max
        self.mask = TimeMasking(self.mask_max)

    def apply(self, img: torch.Tensor, **params):
        return self.mask(img)

    def get_transform_init_args_names(self):
        return {'mask_max': self.mask_max}

