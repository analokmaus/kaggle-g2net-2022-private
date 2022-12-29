import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import FrequencyMasking, TimeMasking
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from scipy.signal import istft, butter, filtfilt
import math
import matplotlib.pyplot as plt
import random
import pickle


'''
Spectromgram
'''
def adaptive_resize(img, img_size, resize_func):
    f, t, ch = img.shape
    resize_f = t // img_size
    t2 = img_size * resize_f
    img = resize_func(
        img[:, :t2, :].reshape(f, t2//resize_f, resize_f, ch), axis=2)
    return img


class ToSpectrogram(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        if np.iscomplexobj(img):
            return img.real ** 2 + img.imag ** 2
        else:
            return img

    def get_transform_init_args_names(self):
        return ()


class NormalizeSpectrogram(ImageOnlyTransform):
    def __init__(self, method='mean', always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        assert method in [
            'mean', 'constant', 'median', 'concat', 'chris', 
            'column_wise', 'row_wise', 'column_row_wise', 'column_wise_sqrt']
        self.method = method

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        if self.method == 'mean':
            for ch in range(img.shape[2]):
                img[:, :, ch] /= img[:, :, ch].mean() 
        elif self.method == 'constant':
            img /= 13.
        elif self.method == 'median':
            for ch in range(img.shape[2]):
                img[:, :, ch] /= np.median(img[:, :, ch])
        elif self.method == 'concat':
            img2 = np.empty((img.shape[0], img.shape[1], 4), dtype=np.float32)
            for ch in range(img.shape[2]):
                img2[:, :, 2*ch] = img[:, :, ch] / img[:, :, ch].mean() 
                img2[:, :, 2*ch+1] = img[:, :, ch] / 13.
            img = img2
        elif self.method == 'chris':
            for ch in range(img.shape[2]):
                img[:, :, ch] /= img[:, :, ch].mean() 
            img -= img.mean()
            img /= img.std()
        elif self.method == 'column_wise':
            for ch in range(img.shape[2]):
                img[:, :, ch] /= img[:, :, ch].mean() 
                img[:, :, ch] -= img[:, :, ch].mean(axis=0)[None, :]
                img[:, :, ch] /= (img[:, :, ch].std(axis=0)[None, :] + 1e-6)
        elif self.method == 'column_wise_sqrt':
            for ch in range(img.shape[2]):
                img[:, :, ch] = np.sqrt(img[:, :, ch])
                img[:, :, ch] -= img[:, :, ch].mean(axis=0)[None, :]
                img[:, :, ch] /= (img[:, :, ch].std(axis=0)[None, :] + 1e-6)
        elif self.method == 'row_wise':
            for ch in range(img.shape[2]):
                img[:, :, ch] /= img[:, :, ch].mean() 
                img[:, :, ch] -= img[:, :, ch].mean(axis=1)[:, None]
                img[:, :, ch] /= (img[:, :, ch].std(axis=1)[:, None] + 1e-6)
        elif self.method == 'column_row_wise':
            for ch in range(img.shape[2]):
                img[:, :, ch] /= img[:, :, ch].mean() 
                img[:, :, ch] -= img[:, :, ch].mean(axis=0)[None, :]
                img[:, :, ch] /= (img[:, :, ch].std(axis=0)[None, :] + 1e-6)
                img[:, :, ch] -= img[:, :, ch].mean(axis=1)[:, None]
                img[:, :, ch] /= (img[:, :, ch].std(axis=1)[:, None] + 1e-6)
        return img

    def get_transform_init_args_names(self):
        return {'method': self.method}
    

class AdaptiveResize(DualTransform):
    def __init__(self, resize_factor=16, img_size=None, method='mean', always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.resize_f = resize_factor
        self.img_size = img_size
        assert method in ['mean', 'max']
        self.method = method
        if self.method == 'mean':
            self.resize_func = np.mean
        elif self.method == 'max':
            self.resize_func = np.max

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        if self.img_size is None:
            img_size = img.shape[1] // self.resize_f
        else:
            img_size = self.img_size
        return adaptive_resize(img, img_size, self.resize_func)

    def apply_to_mask(self, img: np.ndarray, **params):
        if self.img_size is None:
            img_size = img.shape[1] // self.resize_f
        else:
            img_size = self.img_size
        return adaptive_resize(img, img_size, self.resize_func)

    def get_transform_init_args_names(self):
        return {'resize_f': self.resize_f, 'img_size': self.img_size, 'method': self.method}


class Crop352(DualTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        return img[4:356, :, :]

    def apply_to_mask(self, img: np.ndarray, **params):
        return img[4:356, :, :]

    def get_transform_init_args_names(self):
        return {}


class RandomAmplify(ImageOnlyTransform):
    def __init__(self, amp_range=(0.9, 1.1), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.amp_range = amp_range

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        amp = np.random.uniform(*self.amp_range)
        return img * amp

    def get_transform_init_args_names(self):
        return {'amp_range': self.amp_range}


class AddChannel(ImageOnlyTransform):
    def __init__(self, channel='diff', always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        assert channel in ['diff']
        self.channel = channel

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        img = np.concatenate([img, (img[:, :, 0] -img[:, :, 1])[:, :, None]], axis=2)
        return img

    def get_transform_init_args_names(self):
        return {'channel': self.channel}


class DropChannel(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        
    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        img[:, :, np.random.randint(0, 2)] = 0
        return img

    def get_transform_init_args_names(self):
        return {}


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
    def __init__(self, band_width=64, strength=1.0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.band_width = band_width
        self.strength = strength

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        band_width = np.random.randint(0, self.band_width)
        noise_center = np.random.randint(0, img.shape[1]- band_width, size=2)
        strength = np.random.uniform(0, np.median(img) * self.strength)
        gaussian_noise = strength * (
            np.random.normal(0, 1, size=(360, band_width, 2)) ** 2)
        img[:, noise_center[0]:noise_center[0]+band_width, 0] += gaussian_noise[:, :, 0]
        img[:, noise_center[1]:noise_center[1]+band_width, 1] += gaussian_noise[:, :, 1]
        return img

    def get_transform_init_args_names(self):
        return {'band_width': self.band_width, 'strength': self.strength}


class ClipSignal(ImageOnlyTransform):
    def __init__(self, low=0, high=10, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.low = low
        self.high = high

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        return img.clip(self.low, self.high)

    def get_transform_init_args_names(self):
        return {'low': self.low, 'high': self.high}


class InjectTimeNoise(ImageOnlyTransform):
    def __init__(self, noise_file, resize_factor=16, strength=(0.75, 1.0), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.noise_file = noise_file
        with open(noise_file, 'rb') as f:
            self.noise_dict = pickle.load(f)
        self.resize_f = resize_factor
        self.strength = strength
    
    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        tnoise_h1 = random.choice(self.noise_dict['H1']) # (t1,)
        tnoise_l1 = random.choice(self.noise_dict['L1']) # (t2,)
        tnoise_h1 = adaptive_resize(
            np.repeat(tnoise_h1[None, :], img.shape[0] // self.resize_f, axis=0)[:, :, None], self.resize_f, np.mean)[:, :, 0]
        tnoise_l1 = adaptive_resize(
            np.repeat(tnoise_l1[None, :], img.shape[0] // self.resize_f, axis=0)[:, :, None], self.resize_f, np.mean)[:, :, 0]
        noise_std = np.random.uniform(img.std()*0.9, img.std()*1.1)
        noise_scale = np.random.uniform(self.strength[0], self.strength[1])
        tnoise_h1 += np.random.normal(0, noise_std, size=[*tnoise_h1.shape])
        tnoise_h1 -= img[:, :, 0].mean()
        tnoise_l1 += np.random.normal(0, noise_std, size=[*tnoise_l1.shape])
        tnoise_l1 -= img[:, :, 1].mean()
        slice_h1 = np.random.randint(0, tnoise_h1.shape[1]-img.shape[1])
        slice_l1 = np.random.randint(0, tnoise_l1.shape[1]-img.shape[1])
        img[:, :, 0] += (tnoise_h1[:, slice_h1:slice_h1+img.shape[1]] * noise_scale)
        img[:, :, 1] += (tnoise_l1[:, slice_l1:slice_l1+img.shape[1]] * noise_scale)
        return img.clip(0, None)

    def get_transform_init_args_names(self):
        return {'noise_file': self.noise_file, 'resize_f': self.resize_f, 'strength': self.strength}


class InjectAnomaly(ImageOnlyTransform):
    def __init__(self, anomaly_file, detector='H1', always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.anomaly_file = anomaly_file
        with open(anomaly_file, 'rb') as f:
            self.anomaly_dict = pickle.load(f)
        self.detector = detector
    
    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        anomaly, ref_amp = random.choice(self.anomaly_dict[self.detector]) 
        if ref_amp <= 10:
            amp = np.random.uniform(1.5, 10)
        elif 10 < ref_amp <= 100:
            amp = np.random.uniform(10, 100)
        elif ref_amp > 100:
            amp = np.random.uniform(100, 1000)
        anomaly *= amp
        flip_aug_seed = np.random.random()
        if 0.25 < flip_aug_seed <= 0.5:
            anomaly = anomaly[:, ::-1]
        elif 0.5 < flip_aug_seed <= 0.75:
            anomaly = anomaly[::-1, :]
        elif 0.75 < flip_aug_seed:
            anomaly = anomaly[::-1, ::-1]
        if img.shape[0] > anomaly.shape[0]:
            slice_f = np.random.randint(0, img.shape[0]-anomaly.shape[0])
        else:
            slice_f = 0
        if img.shape[1] > anomaly.shape[1]:
            slice_t = np.random.randint(0, img.shape[1]-anomaly.shape[1])
        elif anomaly.shape[1] > img.shape[1]:
            slice_t = 0
            anomaly_slice = np.random.randint(0, anomaly.shape[1]-img.shape[1])
            anomaly = anomaly[:, anomaly_slice:anomaly_slice+img.shape[1]]
        else:
            slice_t = 0
        if self.detector == 'H1':
            img[slice_f:slice_f+anomaly.shape[0], slice_t:slice_t+anomaly.shape[1], 0] += anomaly
        elif self.detector == 'L1':
            img[slice_f:slice_f+anomaly.shape[0], slice_t:slice_t+anomaly.shape[1], 1] += anomaly
        return img

    def get_transform_init_args_names(self):
        return {'anomaly_file': self.anomaly_file, 'detector': self.detector}


class RemoveAnomaly(ImageOnlyTransform):
    def __init__(self, n_sigma=25, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.n_sigma = n_sigma
    
    def apply(self, img: torch.Tensor, **params): # img: (ch, freq, t)
        specs_std = img.std(dim=(1, 2))
        specs_min = img.amin(dim=(1, 2))
        specs_max = img.amax(dim=(1, 2))
        peak_sigma = (specs_max - specs_min) / specs_std
        if peak_sigma.amax() > 25.0:
            img[torch.argmax(peak_sigma)] = 0 # drop single image with anomaly
        return img

    def get_transform_init_args_names(self):
        return {'n_sigma': self.n_sigma}


'''
Waveform
'''
class ToWaveform(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params): # img: (freq, t, ch)
        waveforms = []
        for ch in range(img.shape[2]):
            waveforms.append(istft(img[:, :, ch], nperseg=2)[1].reshape(1, -1)*10)
        return np.stack(waveforms, axis=2) # (1, t, ch)

    def get_transform_init_args_names(self):
        return ()


class WaveToTensor(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img: np.ndarray, **params): # img: (1, t, ch)
        return torch.from_numpy(img).permute(2, 0, 1).squeeze(1).float()

    def get_transform_init_args_names(self):
        return ()


class BandPass(ImageOnlyTransform):
    def __init__(self, 
                 lower=50,
                 upper=500,
                 sr=2048,
                 order=8,
                 always_apply=True, 
                 p=1.0):
        super().__init__(always_apply, p)
        self.lower = lower
        self.upper = upper
        self.sr = sr
        self.order = order
        self._b, self._a = butter(
            self.order, (self.lower, self.upper), btype='bandpass', fs=self.sr)
        
    def apply(self, img: np.ndarray, **params):
        new_img = []
        for ch in range(img.shape[2]):
            new_img.append(filtfilt(self._b, self._a, img[:, :, ch].reshape(-1)).reshape(1, -1))
        return np.stack(new_img, axis=2)

    def get_transform_init_args_names(self):
        return {'lower': self.lower, 'upper': self.upper, 'sr': self.sr}
    