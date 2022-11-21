import numpy as np
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


class ShiftImage(ImageOnlyTransform):
    def __init__(self, x_max=72, y_max=36, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.x_max = x_max
        self.y_max = y_max

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        if self.x_max > 0:
            shift_x = np.random.randint(1, self.x_max)
            img = np.roll(img, shift_x, axis=1)
        if self.y_max > 0:
            shift_y = np.random.randint(1, self.y_max)
            img = np.roll(img, shift_y, axis=0)
        return img

    def get_transform_init_args_names(self):
        return {'x_max': self.x_max, 'y_max': self.y_max}
