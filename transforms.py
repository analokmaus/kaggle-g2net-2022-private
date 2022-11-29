import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import FrequencyMasking, TimeMasking
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
import math
import matplotlib.pyplot as plt


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


class CropImage(ImageOnlyTransform):
    def __init__(self, crop_size=512, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.crop_size = crop_size

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        img = img[:, :self.crop_size, :]
        return img

    def get_transform_init_args_names(self):
        return {'crop_size': self.crop_size}


class RandomCrop(ImageOnlyTransform):
    def __init__(self, crop_size=512, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.crop_size = crop_size

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        img_size = img.shape[1]
        if img_size > self.crop_size:
            slice_start = np.random.randint(0, img_size - self.crop_size)
            img = img[:, slice_start:slice_start+self.crop_size, :]
        return img

    def get_transform_init_args_names(self):
        return {'crop_size': self.crop_size}


class SwapChannel(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        return img[:, :, ::-1].copy()

    def get_transform_init_args_names(self):
        return ()


class MixupChannel(ImageOnlyTransform):
    def __init__(self, num_segments=20, fix_area=False, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.num_segments = num_segments
        self.fix_area = fix_area

    def _make_segment_index(self, length, num_segment=10):
        mixup_indice = np.zeros(length, dtype=np.uint8)
        segment_pos = np.random.randint(0, length-1, num_segment)
        if self.fix_area:
            segment_len = np.random.random(num_segment)
            segment_len = ((segment_len / segment_len.sum()) * length).round().astype(np.uint8)
        else:
            segment_len = np.random.randint(5, length // self.num_segments, num_segment)
        for seg_start, seg_len in zip(segment_pos, segment_len):
            segment_len = length//num_segment
            mixup_indice[seg_start:seg_start+seg_len] = 1
        mixup_indice = mixup_indice.astype(bool)
        return mixup_indice

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        num_segment = np.random.randint(0, self.num_segments)
        if num_segment == 0:
            return img
        else:
            img2 = img.copy()
            mu_idx = self._make_segment_index(img.shape[1])
            img2[:, mu_idx, 0] = img[:, mu_idx, 1]
            img2[:, mu_idx, 1] = img[:, mu_idx, 0]
            return img2
        
    def get_transform_init_args_names(self):
        return {'num_segments': self.num_segments, 'fix_area': self.fix_area}


class MixupChannel2(ImageOnlyTransform):
    def __init__(self, mixup_length=256, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mixup_length = mixup_length

    def _make_segment_index(self, length, num_segment=10):
        mixup_indice = np.zeros(length, dtype=np.uint8)
        segment_pos = np.random.randint(0, length-1, num_segment)
        if self.fix_area:
            segment_len = np.random.random(num_segment)
            segment_len = ((segment_len / segment_len.sum()) * length).round().astype(np.uint8)
        else:
            segment_len = np.random.randint(5, length // self.num_segments, num_segment)
        for seg_start, seg_len in zip(segment_pos, segment_len):
            segment_len = length//num_segment
            mixup_indice[seg_start:seg_start+seg_len] = 1
        mixup_indice = mixup_indice.astype(bool)
        return mixup_indice

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        length = np.random.randint(0, self.mixup_length)
        if length == 0:
            return img
        else:
            img2 = img.copy()
            mu_idx = np.random.choice(np.arange(img.shape[1]), size=length, replace=False)
            img2[:, mu_idx, 0] = img[:, mu_idx, 1]
            img2[:, mu_idx, 1] = img[:, mu_idx, 0]
            return img2
        
    def get_transform_init_args_names(self):
        return {'mixup_length': self.mixup_length}


class DeltaNoise(ImageOnlyTransform):
    def __init__(self, channel='random', strength=(5, 50), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        assert channel in ['random', 'both', 0, 1]
        self.channel = channel
        self.strength = strength

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        noise_center = np.random.randint(0, 359)
        eps = 0.5
        strength = np.random.uniform(*self.strength)
        y_scale = np.random.uniform(2, 10)
        y = np.repeat(np.arange(360).reshape(360, 1), img.shape[1], axis=1)
        y += np.random.randint(-2, 2, size=y.shape)
        noise = eps / (math.pi * (((y-noise_center)/y_scale+1e-4)**2 + eps**2)) * strength
        # noise = eps * np.power(np.abs((y-noise_center)/y_scale+1e-4), eps - 1) * strength

        if self.channel == 'random':
            img[:, :, np.random.randint(0, img.shape[-1]-1)] += noise
        elif self.channel == 'both':
            img += noise[:, :, None]
        else:
            img[:, :, self.channel] += noise
        return img

    def get_transform_init_args_names(self):
        return {'channel': self.channel, 'strength': self.strength}


class BandNoise(ImageOnlyTransform):
    def __init__(self, band_width=64, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.band_width = band_width

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        band_width = np.random.randint(0, self.band_width)
        noise_center = np.random.randint(0, img.shape[1]- band_width, size=2)
        strength = np.random.uniform(0, np.median(img))
        gaussian_noise = strength * (
            np.random.normal(0, 1, size=(360, band_width, 2)) ** 2)
        img[:, noise_center[0]:noise_center[0]+band_width, 0] += gaussian_noise[:, :, 0]
        img[:, noise_center[1]:noise_center[1]+band_width, 1] += gaussian_noise[:, :, 1]
        return img

    def get_transform_init_args_names(self):
        return {'band_width': self.band_width}


class ClipSignal(ImageOnlyTransform):
    def __init__(self, low=0, high=10, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.low = low
        self.high = high

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        return img.clip(self.low, self.high)

    def get_transform_init_args_names(self):
        return {'low': self.low, 'high': self.high}