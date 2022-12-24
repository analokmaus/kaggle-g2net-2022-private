import numpy as np
import pandas as pd
import os
import torch
import torch.utils.data as D
import matplotlib.pyplot as plt
# import h5py
import pickle
from pathlib import Path
from scipy.stats import norm
from scipy import ndimage
import cv2


'''

'''
ARTIFACT_NSIGMA = 6


def laeyoung_normalize(sft):
        sft = np.abs(sft) ** 2
        pos = int(sft.size * 0.99903)
        exp = norm.ppf((pos + 0.4) / (sft.size + 0.215))
        scale = np.partition(sft.flatten(), pos, -1)[pos]
        sft /= (scale / exp.astype(scale.dtype) ** 2)
        return sft


def extract_artifact(spec, n_sigma=8):
    if np.iscomplexobj(spec):
        spec = spec.real**2 + spec.imag**2
    spec_std = spec.std()
    spec_min = spec.min()
    amp_map = (spec - spec_min) / spec_std
    artifact_map = amp_map > n_sigma
    return artifact_map


def reconstruct_from_stat(mean_arr):
    spec = np.zeros((360, len(mean_arr)), dtype=np.float32)
    for t, mean in enumerate(mean_arr):
        spec[:, t] = np.random.chisquare(2, 360)
        factor = mean / spec[:, t].mean()
        spec[:, t] *= factor
    return spec


def reconstruct_from_stat_complex(mean_arr):
    real = np.random.normal(size=(360, len(mean_arr)))
    imag = np.random.normal(size=(360, len(mean_arr)))
    for t, mean in enumerate(mean_arr):
        factor = mean / (real[:, t]**2+imag[:, t]**2).mean()
        real[:, t] *= np.sqrt(factor)
        imag[:, t] *= np.sqrt(factor)
    return real + imag * 1j


'''
Dataset
'''
class G2Net2022Dataset(D.Dataset): # do NOT use
    '''
    '''
    def __init__(
        self, 
        df, 
        data_dir=Path('input/g2net-detecting-continuous-gravitational-waves/train'),
        normalize='none',
        match_time=False,
        fillna=False,
        spec_diff=False,
        random_crop=False, # do Not use
        resize_factor=1,
        resize_method='mean',
        is_test=False,
        transforms=None,
        ):
        self.df = df
        self.data_dir = data_dir
        self.norm = normalize
        self.match = match_time
        self.fillna = fillna
        self.diff = spec_diff
        self.random_crop = random_crop
        self.resize_f = resize_factor
        if resize_method == 'mean':
            self.resize_func = np.mean
        elif resize_method == 'max':
            self.resize_func = np.max
        else:
            raise ValueError(f'{resize_method}')
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img, target = self._load_spec(index)
        return img, target

    def _load_spec(self, index):
        r = self.df.iloc[index]
        target = torch.tensor([r['target']]).float()
        fname = self.data_dir/f'{r.id}.pickle'
        with open(fname, 'rb') as f:
            data = pickle.load(f)[r['id']]
        spec_h1, time_h1 = data['H1']['SFTs']*1e22, data['H1']['timestamps_GPS']
        spec_l1, time_l1 = data['L1']['SFTs']*1e22, data['L1']['timestamps_GPS']
        
        if self.norm == 'laeyoung':
            spec_h1 = laeyoung_normalize(spec_h1)
            spec_l1 = laeyoung_normalize(spec_l1)
        else:
            spec_h1 = spec_h1.real**2 + spec_h1.imag**2
            spec_l1 = spec_l1.real**2 + spec_l1.imag**2
            if self.norm == 'local':
                spec_h1 /= np.mean(spec_h1)
                spec_l1 /= np.mean(spec_l1) 
            elif self.norm == 'global':
                spec_h1 /= 13.
                spec_l1 /= 13.
            elif self.norm == 'local_median':
                spec_h1 /= np.median(spec_h1)
                spec_l1 /= np.median(spec_l1)
            elif self.norm == 'local_25%':
                spec_h1 /= np.percentile(spec_h1, 25)
                spec_l1 /= np.percentile(spec_l1, 25)

        if self.match:
            _spec = np.full((2, 360, 5760), np.nan, np.float32)
            ref_time = min(time_h1.min(), time_l1.min())
            frame_h1 = ((time_h1 - ref_time) / 1800).round().astype(np.uint64)
            frame_l1 = ((time_l1 - ref_time) / 1800).round().astype(np.uint64)
            _spec[0][:, frame_h1[frame_h1 < 5760]] = spec_h1[:, frame_h1 < 5760]
            _spec[1][:, frame_l1[frame_l1 < 5760]] = spec_l1[:, frame_l1 < 5760]
            spec_h1, spec_l1 = _spec[0], _spec[1]
        else:
            if spec_h1.shape[1] < spec_l1.shape[1]:
                spec_l1 = spec_l1[:, :spec_h1.shape[1]]
            elif spec_h1.shape[1] > spec_l1.shape[1]:
                spec_h1 = spec_h1[:, :spec_l1.shape[1]]
        
        if self.fillna:
            spec_h1[np.isnan(spec_h1)] = (np.random.randn(*spec_h1[np.isnan(spec_h1)].shape)**2) * np.mean(spec_h1[~np.isnan(spec_h1)])
            spec_l1[np.isnan(spec_l1)] = (np.random.randn(*spec_l1[np.isnan(spec_l1)].shape)**2) * np.mean(spec_l1[~np.isnan(spec_l1)])
        else:
            spec_h1[np.isnan(spec_h1)] = 0.
            spec_l1[np.isnan(spec_l1)] = 0.
    
        if self.resize_f != 1:
            if self.match:
                spec_h1 = self.resize_func(
                    spec_h1.reshape(360, 5760//self.resize_f, self.resize_f), axis=2)
                spec_l1 = self.resize_func(
                    spec_l1.reshape(360, 5760//self.resize_f, self.resize_f), axis=2)
            else:
                img_size = spec_h1.shape[1]
                img_size2 = int((img_size // self.resize_f) * self.resize_f)
                spec_h1 = self.resize_func(
                    spec_h1[:, :img_size2].reshape(360, img_size2//self.resize_f, self.resize_f), axis=2)
                spec_l1 = self.resize_func(
                    spec_l1[:, :img_size2].reshape(360, img_size2//self.resize_f, self.resize_f), axis=2)

        img = np.stack((spec_h1, spec_l1), axis=2) # (360, t, 2)
        
        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.diff:
            if isinstance(img, torch.Tensor):
                img = torch.concat([img, (img[0] - img[1])[None, :, :]], axis=0) # (3, f, t)
            else:
                img = np.concatenate([img, (img[0] - img[1])[None, :, :]], axis=0) # (3, f, t)
        
        return img, target


class G2Net2022Dataset2(D.Dataset): # Do NOT use
    '''
    '''
    def __init__(
        self, 
        df, 
        data_dir=None, # will be ignored
        normalize='none',
        match_time=False,
        fillna=False,
        spec_diff=False,
        random_crop=False, # do Not use
        resize_factor=1,
        resize_method='mean',
        is_test=False,
        transforms=None,
        ):
        self.df = df
        self.norm = normalize
        self.match = match_time
        self.fillna = fillna
        self.diff = spec_diff
        self.random_crop = random_crop
        self.resize_f = resize_factor
        if resize_method == 'mean':
            self.resize_func = np.mean
        elif resize_method == 'max':
            self.resize_func = np.max
        else:
            raise ValueError(f'{resize_method}')
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img, target = self._load_spec(index)
        return img, target

    def _load_spec(self, index):
        r = self.df.iloc[index]
        target = torch.tensor([r['target']]).float()
        fname = r['path']
        with open(fname, 'rb') as f:
            data = pickle.load(f)[r['id']]
        spec_h1, time_h1 = data['H1']['SFTs']*1e22, data['H1']['timestamps_GPS']
        spec_l1, time_l1 = data['L1']['SFTs']*1e22, data['L1']['timestamps_GPS']
        
        if self.norm == 'local':
            spec_h1 = spec_h1.real**2 + spec_h1.imag**2
            spec_l1 = spec_l1.real**2 + spec_l1.imag**2
            spec_h1 /= np.mean(spec_h1)
            spec_l1 /= np.mean(spec_l1) 
        elif self.norm == 'global':
            spec_h1 = spec_h1.real**2 + spec_h1.imag**2
            spec_l1 = spec_l1.real**2 + spec_l1.imag**2
            spec_h1 /= 13.
            spec_l1 /= 13.
        elif self.norm == 'local_median':
            spec_h1 = spec_h1.real**2 + spec_h1.imag**2
            spec_l1 = spec_l1.real**2 + spec_l1.imag**2
            spec_h1 /= np.median(spec_h1)
            spec_l1 /= np.median(spec_l1)
        elif self.norm == 'local_25%':
            spec_h1 = spec_h1.real**2 + spec_h1.imag**2
            spec_l1 = spec_l1.real**2 + spec_l1.imag**2
            spec_h1 /= np.percentile(spec_h1, 25)
            spec_l1 /= np.percentile(spec_l1, 25)
        elif self.norm == 'laeyoung':
            spec_h1 = laeyoung_normalize(spec_h1)
            spec_l1 = laeyoung_normalize(spec_l1)
        else:
            pass

        if self.match:
            _spec = np.full((2, 360, 5760), np.nan, np.float32)
            ref_time = min(time_h1.min(), time_l1.min())
            frame_h1 = ((time_h1 - ref_time) / 1800).round().astype(np.uint64)
            frame_l1 = ((time_l1 - ref_time) / 1800).round().astype(np.uint64)
            _spec[0][:, frame_h1[frame_h1 < 5760]] = spec_h1[:, frame_h1 < 5760]
            _spec[1][:, frame_l1[frame_l1 < 5760]] = spec_l1[:, frame_l1 < 5760]
            spec_h1, spec_l1 = _spec[0], _spec[1]
        else:
            if spec_h1.shape[1] < spec_l1.shape[1]:
                spec_l1 = spec_l1[:, :spec_h1.shape[1]]
            elif spec_h1.shape[1] > spec_l1.shape[1]:
                spec_h1 = spec_h1[:, :spec_l1.shape[1]]
        
        if self.fillna:
            spec_h1[np.isnan(spec_h1)] = (np.random.randn(*spec_h1[np.isnan(spec_h1)].shape)**2) * np.mean(spec_h1[~np.isnan(spec_h1)])
            spec_l1[np.isnan(spec_l1)] = (np.random.randn(*spec_l1[np.isnan(spec_l1)].shape)**2) * np.mean(spec_l1[~np.isnan(spec_l1)])
        else:
            spec_h1[np.isnan(spec_h1)] = 0.
            spec_l1[np.isnan(spec_l1)] = 0.
    
        if self.resize_f != 1:
            if self.match:
                spec_h1 = self.resize_func(
                    spec_h1.reshape(360, 5760//self.resize_f, self.resize_f), axis=2)
                spec_l1 = self.resize_func(
                    spec_l1.reshape(360, 5760//self.resize_f, self.resize_f), axis=2)
            else:
                if (not self.is_test) and self.random_crop:
                    slice_start = np.random.randint(0, spec_h1.shape[1] - 4096)
                    spec_h1 = self.resize_func(
                        spec_h1[:, slice_start:slice_start+4096].reshape(360, 4096//self.resize_f, self.resize_f), axis=2)
                    spec_l1 = self.resize_func(
                        spec_l1[:, slice_start:slice_start+4096].reshape(360, 4096//self.resize_f, self.resize_f), axis=2)
                else:
                    img_size = spec_h1.shape[1]
                    img_size2 = int((img_size // self.resize_f) * self.resize_f)
                    spec_h1 = self.resize_func(
                        spec_h1[:, :img_size2].reshape(360, img_size2//self.resize_f, self.resize_f), axis=2)
                    spec_l1 = self.resize_func(
                        spec_l1[:, :img_size2].reshape(360, img_size2//self.resize_f, self.resize_f), axis=2)
        
        img = np.stack((spec_h1, spec_l1), axis=2) # (360, t, 2)
        
        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.diff:
            img = torch.concat([img, (img[0] - img[1])[None, :, :]], axis=0) # (3, f, t)
        else:
                img = np.concatenate([img, (img[0] - img[1])[None, :, :]], axis=0) # (3, f, t)
        
        return img, target


class G2Net2022Dataset3(D.Dataset):
    '''
    '''
    def __init__(
        self, 
        df, 
        data_dir=Path('input/g2net-detecting-continuous-gravitational-waves/train'),
        match_time=False,
        fillna=False,
        is_test=False,
        preprocess=None,
        transforms=None,
        cache_limit=0, # in GB
        ):
        self.df = df
        self.data_dir = data_dir
        self.match = match_time
        self.fillna = fillna
        self.preprocess = preprocess
        self.transforms = transforms
        self.is_test = is_test
        self.cache = {'size': 0}
        self.cache_limit = cache_limit

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img, target = self._load_spec(index)
        return img, target

    def _load_spec(self, index):
        r = self.df.iloc[index]
        target = torch.tensor([r['target']]).float()
        if r['id'] in self.cache.keys():
            img = self.cache[r['id']]
        else:
            if 'path' in self.df.columns:
                fname = r['path']
            else:
                fname = self.data_dir/f'{r.id}.pickle'
            
            with open(fname, 'rb') as f:
                data = pickle.load(f)

            if 'H1' in data.keys(): # simple format
                spec_h1, time_h1 = data['H1']['spectrogram'].astype(np.float32), data['H1']['timestamps']
                spec_l1, time_l1 = data['L1']['spectrogram'].astype(np.float32), data['L1']['timestamps']
            else: # original format
                gid = list(data.keys())[0]
                data = data[gid]
                spec_h1, time_h1 = data['H1']['SFTs']*1e22, data['H1']['timestamps_GPS']
                spec_l1, time_l1 = data['L1']['SFTs']*1e22, data['L1']['timestamps_GPS']
                spec_h1 = spec_h1.real**2 + spec_h1.imag**2
                spec_l1 = spec_l1.real**2 + spec_l1.imag**2

            if self.match:
                _spec = np.full((2, 360, 5760), 0., np.float32)
                ref_time = min(time_h1.min(), time_l1.min())
                frame_h1 = ((time_h1 - ref_time) / 1800).round().astype(np.uint64)
                frame_l1 = ((time_l1 - ref_time) / 1800).round().astype(np.uint64)
                _spec[0][:, frame_h1[frame_h1 < 5760]] = spec_h1[:, frame_h1 < 5760]
                _spec[1][:, frame_l1[frame_l1 < 5760]] = spec_l1[:, frame_l1 < 5760]
                spec_h1, spec_l1 = _spec[0], _spec[1]
            else:
                if spec_h1.shape[1] < spec_l1.shape[1]:
                    spec_l1 = spec_l1[:, :spec_h1.shape[1]]
                elif spec_h1.shape[1] > spec_l1.shape[1]:
                    spec_h1 = spec_h1[:, :spec_l1.shape[1]]

            img = np.stack((spec_h1, spec_l1), axis=2) # (360, t, 2)

            if self.preprocess:
                img = self.preprocess(image=img)['image']
            
            if self.cache['size'] < self.cache_limit:
                self.cache[r['id']] = img
                self.cache['size'] += img.nbytes / (1024 ** 3)
            else:
                pass

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        return img, target


class ChrisDataset(D.Dataset):
    '''
    '''
    def __init__(
        self, 
        df, 
        data_dir=Path('input/g2net-detecting-continuous-gravitational-waves/train'),
        match_time=False,
        img_size=720,
        max_size=6000,
        is_test=False,
        preprocess=None, # Ignore
        transforms=None,
        cache_limit=0, # in GB
        ):
        self.df = df
        self.data_dir = data_dir
        self.match = match_time
        self.img_size = img_size
        self.max_size = max_size
        self.preprocess = preprocess
        self.transforms = transforms
        self.is_test = is_test
        self.cache = {'size': 0}
        self.cache_limit = cache_limit

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self._load_spec(index)

    def _load_spec(self, index):
        r = self.df.iloc[index]
        target = torch.tensor([r['target']]).float()
        if r['id'] in self.cache.keys():
            img, freq = self.cache[r['id']]
        else:
            if 'path' in self.df.columns:
                fname = r['path']
            else:
                fname = self.data_dir/f'{r.id}.pickle'
            with open(fname, 'rb') as f:
                data = pickle.load(f)
                gid = list(data.keys())[0]
                data = data[gid]
            spec_h1, time_h1 = data['H1']['SFTs']*1e22, data['H1']['timestamps_GPS']
            spec_l1, time_l1 = data['L1']['SFTs']*1e22, data['L1']['timestamps_GPS']
            freq = data['frequency_Hz'][0]

            if self.match:
                _spec = np.full((2, 360, 5760), 0., np.complex64)
                ref_time = min(time_h1.min(), time_l1.min())
                frame_h1 = ((time_h1 - ref_time) / 1800).round().astype(np.uint64)
                frame_l1 = ((time_l1 - ref_time) / 1800).round().astype(np.uint64)
                _spec[0][:, frame_h1[frame_h1 < 5760]] = spec_h1[:, frame_h1 < 5760]
                _spec[1][:, frame_l1[frame_l1 < 5760]] = spec_l1[:, frame_l1 < 5760]
                spec_h1, spec_l1 = _spec[0], _spec[1]
            else:
                if spec_h1.shape[1] < spec_l1.shape[1]:
                    spec_l1 = spec_l1[:, :spec_h1.shape[1]]
                elif spec_h1.shape[1] > spec_l1.shape[1]:
                    spec_h1 = spec_h1[:, :spec_l1.shape[1]]

            img = np.zeros((360, self.img_size, 2), dtype=np.float32)
            for ch, spec in enumerate([spec_h1, spec_l1]):
                across = int(spec.shape[1] / self.img_size)
                across = min(across, int(self.max_size / self.img_size))
                spec = spec[:, :(across*self.img_size)].real**2 + spec[:, :(across*self.img_size)].imag**2
                spec /= np.mean(spec)  # normalize
                # p = wiener(p, (3, 13))
                spec = np.mean(spec.reshape(360, self.img_size, across), axis=2)
                img[:, :, ch] = spec

            if self.cache['size'] < self.cache_limit:
                self.cache[r['id']] = (img, freq)
                self.cache['size'] += (img.nbytes) / (1024 ** 3)
            else:
                pass

        mean0 = img[:, :, 0].mean()
        std0 = img[:, :, 0].std()
        mean1 = img[:, :, 1].mean()
        std1 = img[:, :, 1].std()
        img = img - img.mean() 
        img = img / img.std()

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, torch.FloatTensor([freq]), \
            torch.FloatTensor([mean0]), torch.FloatTensor([std0]), \
                torch.FloatTensor([mean1]), torch.FloatTensor([std1]), target


class G2Net2022Dataset8(D.Dataset):
    '''
    Infinite training set noise based sampling
    '''
    def __init__(
        self, 
        df, 
        data_dir=None,
        signal_path=None,
        signal_dir=None,
        positive_p=0.66,
        noise_mixup_p=0.0,
        dataset_multiplier=1,
        signal_amplifier=1.0,
        match_time=False,
        fillna=False,
        is_test=False,
        preprocess=None,
        transforms=None,
        return_mask=None,
        random_state=0,
        cache_limit=0, # in GB
        ):
        if dataset_multiplier > 1 and not is_test:
            self.df = pd.concat([df] * dataset_multiplier, axis=0).reset_index(drop=True)
        else:
            self.df = df
        self.signal_df = pd.read_csv(signal_path)
        self.data_dir = data_dir
        self.signal_dir = signal_dir
        self.positive_p = positive_p
        self.noise_mixup_p = noise_mixup_p
        self.signal_amp = signal_amplifier
        self.match = match_time
        self.fillna = fillna
        self.preprocess = preprocess
        self.transforms = transforms
        self.is_test = is_test
        self.cache = {'size': 0}
        self.cache_limit = cache_limit
        self.return_mask = return_mask
        np.random.RandomState(random_state)
        np.random.seed(random_state)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self._load_noise_and_signal(index)

    def _load_noise(self, index):
        r = self.df.iloc[index]
        if r['id'] in self.cache.keys():
            img, freq, frame_h1, frame_l1 = self.cache[r['id']]
        else:
            if 'path' in self.df.columns:
                fname = r['path']
            else:
                fname = self.data_dir/f'{r.id}.pickle'
            with open(fname, 'rb') as f:
                data = pickle.load(f)
            
            if 'H1' in data.keys(): # simple format
                spec_h1, time_h1 = data['H1']['spectrogram'].astype(np.float32), data['H1']['timestamps']
                spec_l1, time_l1 = data['L1']['spectrogram'].astype(np.float32), data['L1']['timestamps']
                freq = data['frequency'].mean()
            else: # original format
                gid = list(data.keys())[0]
                data = data[gid]
                spec_h1, time_h1 = data['H1']['SFTs']*1e22, data['H1']['timestamps_GPS']
                spec_l1, time_l1 = data['L1']['SFTs']*1e22, data['L1']['timestamps_GPS']
                spec_h1 = spec_h1.real**2 + spec_h1.imag**2
                spec_l1 = spec_l1.real**2 + spec_l1.imag**2
                freq = data['frequency_Hz'].mean()
            
            ref_time = min(time_h1.min(), time_l1.min())
            frame_h1 = ((time_h1 - ref_time) / 1800).round().astype(np.uint64)
            frame_l1 = ((time_l1 - ref_time) / 1800).round().astype(np.uint64)

            if self.match:
                _spec = np.full((2, 360, 5760), 0., np.float32)
                _spec[0][:, frame_h1[frame_h1 < 5760]] = spec_h1[:, frame_h1 < 5760]
                _spec[1][:, frame_l1[frame_l1 < 5760]] = spec_l1[:, frame_l1 < 5760]
                spec_h1, spec_l1 = _spec[0], _spec[1]
            else:
                if spec_h1.shape[1] < spec_l1.shape[1]:
                    frame_l1 = frame_l1[:spec_h1.shape[1]]
                    spec_l1 = spec_l1[:, :spec_h1.shape[1]]
                elif spec_h1.shape[1] > spec_l1.shape[1]:
                    frame_h1 = frame_h1[:spec_l1.shape[1]]
                    spec_h1 = spec_h1[:, :spec_l1.shape[1]]

            img = np.stack((spec_h1, spec_l1), axis=2) # (360, t, 2)

            if self.cache['size'] < self.cache_limit:
                self.cache[r['id']] = (img, freq, frame_h1, frame_l1)
                self.cache['size'] += img.nbytes / (1024 ** 3)
            else:
                pass

        return img, freq, frame_h1, frame_l1

    def _load_noise_and_signal(self, index):
        r = self.df.iloc[index]
        img, freq, frame_h1, frame_l1 = self._load_noise(index)
        if (np.random.random() < self.noise_mixup_p) and not self.is_test:
            index2 = np.random.randint(0, len(self.df))
            img2, _, _, _ = self._load_noise(index2)
            if img2.shape[1] > img.shape[1]:
                img = img / 2 + img2[:, :img.shape[1], :] / 2
            else:
                img[:, :img2.shape[1], :] -= (img[:, :img2.shape[1], :] / 2 - img2 / 2)

        if (np.random.random() < self.positive_p) and not self.is_test: # inject signal
            cand_signal = self.signal_df.query(f'{freq-50} <= F0 <= {freq+50}').sample()
            sid = cand_signal['id'].values[0]
            with open(self.signal_dir/f'{sid}.pickle', 'rb') as f:
                data = pickle.load(f)
            sft_s, _ = data['sft']*1e22, data['timestamps']
            spec_s = sft_s.real ** 2 + sft_s.imag ** 2
            shift_y = np.random.randint(-110, 110)
            spec_s = np.roll(spec_s, shift_y, axis=0)
            spec_s *= self.signal_amp
            # img[:, :, :] = 0

            if self.match:
                img[:, frame_h1[frame_h1 < 5760], 0] += spec_s[:360, frame_h1[frame_h1 < 5760]]
                img[:, frame_l1[frame_l1 < 5760], 1] += spec_s[:360, frame_l1[frame_l1 < 5760]]
            else:
                img[:, :, 0] += spec_s[:360, frame_h1]
                img[:, :, 1] += spec_s[:360, frame_l1]
            target = torch.tensor([1]).float()
        else:
            spec_s = np.zeros_like(img, dtype=np.float32)
            target = torch.tensor([r['target']]).float()

        if self.preprocess:
            if self.return_mask:
                t = self.preprocess(image=img, mask=spec_s[:360, frame_h1, None])
                img = t['image']
                mask = t['mask']
            else:
                t = self.preprocess(image=img)
                img = t['image']
        
        if self.transforms:
            if self.return_mask:
                t = self.transforms(image=img, mask=mask)
                img = t['image']
                mask = t['mask']
            else:
                t = self.transforms(image=img)
                img = t['image']

        if self.return_mask:
            return img, mask, target
        else:
            return img, target


class G2Net2022Dataset88(D.Dataset):
    '''
    Infinite training set signal based sampling
    '''
    def __init__(
        self, 
        df, 
        data_dir=None,
        noise_path=None,
        noise_dir=None,
        positive_p=0.66,
        noise_mixup_p=0.0,
        signal_amplifier=1.0,
        match_time=False,
        fillna=False,
        is_test=False,
        shift_range=(-110, 110),
        preprocess=None,
        transforms=None,
        return_mask=None,
        random_state=0,
        cache_limit=0, # in GB
        ):
        self.df = df
        self.noise_df = pd.read_csv(noise_path)
        self.data_dir = data_dir
        self.noise_dir = noise_dir
        self.positive_p = positive_p
        self.noise_mixup_p = noise_mixup_p
        self.match = match_time
        self.fillna = fillna
        self.shift_range = shift_range
        self.preprocess = preprocess
        self.transforms = transforms
        self.is_test = is_test
        self.cache = {'size': 0}
        self.cache_limit = cache_limit
        self.return_mask = return_mask
        if isinstance(signal_amplifier, (int, float)):
            self.signal_amp = [signal_amplifier]
        else:
            self.signal_amp = signal_amplifier
        self._epoch = 0
        np.random.RandomState(random_state)
        np.random.seed(random_state)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self._load_noise_and_signal(index)

    def _load_noise(self, r):
        if r['id'] in self.cache.keys():
            img, freq, frame_h1, frame_l1 = self.cache[r['id']]
        else:
            if 'path' in r.index:
                fname = r['path']
            else:
                fname = self.data_dir/f'{r.id}.pickle'
            with open(fname, 'rb') as f:
                data = pickle.load(f)

            if 'H1' in data.keys(): # simple format
                spec_h1, time_h1 = data['H1']['spectrogram'].astype(np.float32), data['H1']['timestamps']
                spec_l1, time_l1 = data['L1']['spectrogram'].astype(np.float32), data['L1']['timestamps']
                freq = data['frequency'].mean()
            else: # original format
                gid = list(data.keys())[0]
                data = data[gid]
                spec_h1, time_h1 = data['H1']['SFTs']*1e22, data['H1']['timestamps_GPS']
                spec_l1, time_l1 = data['L1']['SFTs']*1e22, data['L1']['timestamps_GPS']
                spec_h1 = spec_h1.real**2 + spec_h1.imag**2
                spec_l1 = spec_l1.real**2 + spec_l1.imag**2
                freq = data['frequency_Hz'].mean()

            ref_time = min(time_h1.min(), time_l1.min())
            frame_h1 = ((time_h1 - ref_time) / 1800).round().astype(np.uint64)
            frame_l1 = ((time_l1 - ref_time) / 1800).round().astype(np.uint64)

            if self.match:
                _spec = np.full((2, 360, 5760), 0., np.float32)
                _spec[0][:, frame_h1[frame_h1 < 5760]] = spec_h1[:, frame_h1 < 5760]
                _spec[1][:, frame_l1[frame_l1 < 5760]] = spec_l1[:, frame_l1 < 5760]
                spec_h1, spec_l1 = _spec[0], _spec[1]
            else:
                if spec_h1.shape[1] < spec_l1.shape[1]:
                    frame_l1 = frame_l1[:spec_h1.shape[1]]
                    spec_l1 = spec_l1[:, :spec_h1.shape[1]]
                elif spec_h1.shape[1] > spec_l1.shape[1]:
                    frame_h1 = frame_h1[:spec_l1.shape[1]]
                    spec_h1 = spec_h1[:, :spec_l1.shape[1]]

            img = np.stack((spec_h1, spec_l1), axis=2) # (360, t, 2)

            if self.cache['size'] < self.cache_limit:
                self.cache[r['id']] = (img, freq, frame_h1, frame_l1)
                self.cache['size'] += img.nbytes / (1024 ** 3)
            else:
                pass

        return img, freq, frame_h1, frame_l1

    def _load_signal(self, r):
        if 'path' in r.index:
            fname = r['path']
        else:
            fname = self.data_dir/f'{r.id}.pickle'
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        sft_s, _ = data['sft']*1e22, data['timestamps']
        spec_s = sft_s.real ** 2 + sft_s.imag ** 2
        shift_y = np.random.randint(*self.shift_range)
        spec_s = np.roll(spec_s, shift_y, axis=0)
        if shift_y > 0:
            spec_s[:shift_y, :] = 0
        else:
            spec_s[shift_y:, :] = 0
        spec_s *= self.signal_amp[min(self._epoch, len(self.signal_amp)-1)]
        return spec_s

    def _inject_signal(self, noise, signal, frame_h1, frame_l1):
        # align timestamps
        if self.match:
            noise[:, frame_h1[frame_h1 < 5760], 0] += signal[:360, frame_h1[frame_h1 < 5760]]
            noise[:, frame_l1[frame_l1 < 5760], 1] += signal[:360, frame_l1[frame_l1 < 5760]]
            signal_mask = np.zeros((noise.shape[0], noise.shape[1]), dtype=np.float32)
            if self.return_mask:
                signal_bin = ((signal - signal.min()) / (signal.max() - signal.min()) > 0.25).astype(np.float32)   
                signal_mask[:, frame_h1[frame_h1 < 5760]] = signal_bin[:360, frame_h1[frame_h1 < 5760]]
                signal_mask[:, frame_l1[frame_l1 < 5760]] = signal_bin[:360, frame_l1[frame_l1 < 5760]]
        else:
            noise[:, :, 0] += signal[:360, frame_h1]
            noise[:, :, 1] += signal[:360, frame_l1]
            signal_mask = np.zeros((noise.shape[0], noise.shape[1]), dtype=np.float32)
            if self.return_mask:
                signal_bin = ((signal - signal.min()) / (signal.max() - signal.min()) > 0.25).astype(np.float32)   
                signal_mask[:, :] = signal_bin[:, frame_h1]
                signal_mask[:, :] = signal_bin[:, frame_l1]
        signal_mask[signal_mask!=signal_mask] = 0
        return noise, signal_mask[:, :, None]

    def _load_noise_and_signal(self, index):
        if self.is_test: # test
            r = self.df.iloc[index]
            img, _, _, _ = self._load_noise(r)
            signal_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
            target = torch.tensor([r['target']]).float()

        else: # train
            if np.random.random() < self.positive_p:
                signal_r = self.df.iloc[index]
                signal_freq = signal_r['F0']
                signal_spec = self._load_signal(signal_r)
                
                nearby_noises = self.noise_df.query(f'{signal_freq-50} <= freq <= {signal_freq+50}')
                noise_r = nearby_noises.sample().iloc[0]
                noise_spec, _, frame_h1, frame_l1 = self._load_noise(noise_r)

                if np.random.random() < self.noise_mixup_p: # mixup noise
                    noise_r2 = nearby_noises.sample().iloc[0]
                    noise_spec2, _, _, _ = self._load_noise(noise_r2) # TODO: how to mixup timestamps?
                    if noise_spec2.shape[1] > noise_spec.shape[1]:
                        noise_spec = noise_spec / 2 + noise_spec2[:, :noise_spec.shape[1], :] / 2
                    else:
                        noise_spec[:, :noise_spec2.shape[1], :] -= (noise_spec[:, :noise_spec2.shape[1], :] / 2 - noise_spec2 / 2)

                img, signal_mask = self._inject_signal(noise_spec, signal_spec, frame_h1, frame_l1)
                target = torch.tensor([1]).float()
            else:
                noise_r = self.noise_df.sample().iloc[0]
                noise_freq = noise_r['freq']
                nearby_noises = self.noise_df.query(f'{noise_freq-50} <= freq <= {noise_freq+50}')
                noise_spec, _, frame_h1, frame_l1 = self._load_noise(noise_r)

                if np.random.random() < self.noise_mixup_p: # mixup noise
                    noise_r2 = nearby_noises.sample().iloc[0]
                    noise_spec2, _, _, _ = self._load_noise(noise_r2) # TODO: how to mixup timestamps?
                    if noise_spec2.shape[1] > noise_spec.shape[1]:
                        noise_spec = noise_spec / 2 + noise_spec2[:, :noise_spec.shape[1], :] / 2
                    else:
                        noise_spec[:, :noise_spec2.shape[1], :] -= (noise_spec[:, :noise_spec2.shape[1], :] / 2 - noise_spec2 / 2)
                
                img = noise_spec
                signal_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
                target = torch.tensor([0]).float()

        if self.preprocess:
            if self.return_mask:
                t = self.preprocess(image=img, mask=signal_mask)
                img = t['image']
                signal_mask = t['mask']
            else:
                t = self.preprocess(image=img)
                img = t['image']
        
        if self.transforms:
            if self.return_mask:
                t = self.transforms(image=img, mask=signal_mask)
                img = t['image']
                signal_mask = t['mask']
            else:
                t = self.transforms(image=img)
                img = t['image']

        if self.return_mask:
            return img, signal_mask, target
        else:
            return img, target

    def step(self):
        self._epoch += 1


class G2Net2022Dataset888(D.Dataset):
    '''
    Infinite training set signal based sampling with synthetic noise
    '''
    def __init__(
        self, 
        df, 
        data_dir=None,
        test_stat=None,
        test_dir=None,
        positive_p=0.66,
        signal_amplifier=1.0,
        match_time=False,
        fillna=False,
        is_test=False,
        shift_range=(-150, 150),
        rotate_range=(0, 0),
        preprocess=None,
        transforms=None,
        return_mask=None,
        random_state=0,
        cache_limit=0, # in GB
        ):
        self.df = df
        self.data_dir = data_dir
        with open(test_stat, 'rb') as f:
            self.test_stat = pickle.load(f)
        self.test_dir = test_dir
        self.positive_p = positive_p
        self.match = match_time
        self.fillna = fillna
        self.shift_range = shift_range
        self.rotate_range = rotate_range
        self.preprocess = preprocess
        self.transforms = transforms
        self.is_test = is_test
        self.cache = {'size': 0}
        self.cache_limit = cache_limit
        self.return_mask = return_mask
        if isinstance(signal_amplifier, (int, float)):
            self.signal_amp = [signal_amplifier]
        else:
            self.signal_amp = signal_amplifier
        self._epoch = 0
        np.random.RandomState(random_state)
        np.random.seed(random_state)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self._load_noise_and_signal(index)

    def _load_real_noise(self, r):
        if r['id'] in self.cache.keys() and self.is_test:
            spec_h1, spec_l1, time_h1, time_l1 = self.cache[r['id']]
        else:
            if 'path' in r.index:
                fname = r['path']
            else:
                if self.is_test:
                    fname = self.data_dir/f'{r.id}.pickle'
                else: # in training always read test file
                    fname = self.test_dir/f'{r.base_id}.pickle'

            with open(fname, 'rb') as f:
                data = pickle.load(f)

            if 'H1' in data.keys(): # simple format
                spec_h1, time_h1 = data['H1']['spectrogram'].astype(np.float32), data['H1']['timestamps']
                spec_l1, time_l1 = data['L1']['spectrogram'].astype(np.float32), data['L1']['timestamps']
            else: # original format
                gid = list(data.keys())[0]
                data = data[gid]
                spec_h1, time_h1 = data['H1']['SFTs']*1e22, data['H1']['timestamps_GPS']
                spec_l1, time_l1 = data['L1']['SFTs']*1e22, data['L1']['timestamps_GPS']
                spec_h1 = spec_h1.real**2 + spec_h1.imag**2
                spec_l1 = spec_l1.real**2 + spec_l1.imag**2

            if self.cache['size'] < self.cache_limit and self.is_test:
                self.cache[r['id']] = (spec_h1, spec_l1, time_h1, time_l1)
                self.cache['size'] += (spec_h1.nbytes + spec_l1.nbytes) / (1024 ** 3)
            else:
                pass

        return spec_h1, spec_l1, time_h1, time_l1

    def _genereta_noise(self, r):
        test_stat = self.test_stat[r['base_id']]
        asd_time_h1, asd_freq_h1, time_h1 = test_stat['H1']
        asd_time_l1, asd_freq_l1, time_l1 = test_stat['L1']
        if asd_time_h1.max() < 5 and asd_freq_h1.max() < 5:
            spec_gen_h1 = reconstruct_from_stat(asd_time_h1.clip(0, 5))
        else:
            spec_gen_h1 = reconstruct_from_stat(asd_time_h1.clip(0, 5))
            spec_org = self._load_real_noise(r)[0]
            artifact_map = extract_artifact(spec_org, ARTIFACT_NSIGMA)
            spec_gen_h1[np.where(artifact_map)] = spec_org[np.where(artifact_map)]

        if asd_time_l1.max() < 5 and asd_freq_l1.max() < 5:
            spec_gen_l1 = reconstruct_from_stat(asd_time_l1.clip(0, 5))
        else:
            spec_gen_l1 = reconstruct_from_stat(asd_time_l1.clip(0, 5))
            spec_org = self._load_real_noise(r)[1]
            artifact_map = extract_artifact(spec_org, ARTIFACT_NSIGMA)
            spec_gen_l1[np.where(artifact_map)] = spec_org[np.where(artifact_map)]

        return spec_gen_h1, spec_gen_l1, time_h1, time_l1

    def _preprocess_data(self, r):
        if self.is_test:
            spec_h1, spec_l1, time_h1, time_l1 = self._load_real_noise(r)
        else:
            spec_h1, spec_l1, time_h1, time_l1 = self._genereta_noise(r)

        ref_time = min(time_h1.min(), time_l1.min())
        frame_h1 = ((time_h1 - ref_time) / 1800).round().astype(np.uint64)
        frame_l1 = ((time_l1 - ref_time) / 1800).round().astype(np.uint64)

        if self.match:
            _spec = np.full((2, 360, 5760), 0., np.float32)
            _spec[0][:, frame_h1[frame_h1 < 5760]] = spec_h1[:, frame_h1 < 5760]
            _spec[1][:, frame_l1[frame_l1 < 5760]] = spec_l1[:, frame_l1 < 5760]
            spec_h1, spec_l1 = _spec[0], _spec[1]
        else:
            if spec_h1.shape[1] < spec_l1.shape[1]:
                frame_l1 = frame_l1[:spec_h1.shape[1]]
                spec_l1 = spec_l1[:, :spec_h1.shape[1]]
            elif spec_h1.shape[1] > spec_l1.shape[1]:
                frame_h1 = frame_h1[:spec_l1.shape[1]]
                spec_h1 = spec_h1[:, :spec_l1.shape[1]]

        img = np.stack((spec_h1, spec_l1), axis=2) # (360, t, 2)

        return img, frame_h1, frame_l1

    def _load_signal(self, r):
        if r['id'] in self.cache.keys():
            spec_s = self.cache[r['id']]
        else:
            if 'path' in r.index:
                fname = r['path']
            else:
                fname = self.data_dir/f'{r.id}.pickle'
            with open(fname, 'rb') as f:
                data = pickle.load(f)
            sft_s, _ = data['sft']*1e22, data['timestamps']
            spec_s = sft_s.real**2 + sft_s.imag**2

            if self.cache['size'] < self.cache_limit:
                self.cache[r['id']] = spec_s
                self.cache['size'] += spec_s.nbytes / (1024 ** 3)
            else:
                pass
        
        shift_y = np.random.randint(*self.shift_range)
        spec_s = np.roll(spec_s, shift_y, axis=0)
        if shift_y > 0:
            spec_s[:shift_y, :] = 0
        else:
            spec_s[shift_y:, :] = 0
        if self.rotate_range[0] != 0 or self.rotate_range[0] != 0:
            rotate_ang = np.random.randint(*self.rotate_range)
            spec_s = ndimage.rotate(spec_s, rotate_ang, reshape=False)
        spec_s *= self.signal_amp[min(self._epoch, len(self.signal_amp)-1)]
        return spec_s

    def _inject_signal(self, noise, signal, frame_h1, frame_l1):
        # align timestamps
        if self.match:
            noise[:, frame_h1[frame_h1 < 5760], 0] += signal[:360, frame_h1[frame_h1 < 5760]]
            noise[:, frame_l1[frame_l1 < 5760], 1] += signal[:360, frame_l1[frame_l1 < 5760]]
            signal_mask = np.zeros((noise.shape[0], noise.shape[1]), dtype=np.float32)
            if self.return_mask:
                signal_bin = ((signal - signal.min()) / (signal.max() - signal.min()) > 0.25).astype(np.float32)   
                signal_mask[:, frame_h1[frame_h1 < 5760]] = signal_bin[:360, frame_h1[frame_h1 < 5760]]
                signal_mask[:, frame_l1[frame_l1 < 5760]] = signal_bin[:360, frame_l1[frame_l1 < 5760]]
        else:
            noise[:, :, 0] += signal[:360, frame_h1]
            noise[:, :, 1] += signal[:360, frame_l1]
            signal_mask = np.zeros((noise.shape[0], noise.shape[1]), dtype=np.float32)
            if self.return_mask:
                signal_bin = ((signal - signal.min()) / (signal.max() - signal.min()) > 0.25).astype(np.float32)   
                signal_mask[:, :] = signal_bin[:, frame_h1]
                signal_mask[:, :] = signal_bin[:, frame_l1]
        signal_mask[signal_mask!=signal_mask] = 0
        return noise, signal_mask[:, :, None]

    def _load_noise_and_signal(self, index):
        if self.is_test: # test
            r = self.df.iloc[index]
            img, _, _ = self._preprocess_data(r)
            signal_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
            target = torch.tensor([r['target']]).float()

        else: # train
            if np.random.random() < self.positive_p:
                signal_r = self.df.iloc[index]
                signal_spec = self._load_signal(signal_r)
                noise_spec, frame_h1, frame_l1 = self._preprocess_data(signal_r)

                img, signal_mask = self._inject_signal(noise_spec, signal_spec, frame_h1, frame_l1)
                target = torch.tensor([1]).float()
            else:
                signal_r = self.df.iloc[index]
                noise_spec, frame_h1, frame_l1 = self._preprocess_data(signal_r)

                img = noise_spec
                signal_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
                target = torch.tensor([0]).float()

        if self.preprocess:
            if self.return_mask:
                t = self.preprocess(image=img, mask=signal_mask)
                img = t['image']
                signal_mask = t['mask']
            else:
                t = self.preprocess(image=img)
                img = t['image']
        
        if self.transforms:
            if self.return_mask:
                t = self.transforms(image=img, mask=signal_mask)
                img = t['image']
                signal_mask = t['mask']
            else:
                t = self.transforms(image=img)
                img = t['image']

        if self.return_mask:
            return img, signal_mask, target
        else:
            return img, target

    def step(self):
        self._epoch += 1


class G2Net2022Dataset8888(D.Dataset):
    '''
    Infinite training set signal based sampling with synthetic noise (re/im blend)
    '''
    def __init__(
        self, 
        df, 
        data_dir=None,
        test_stat=None,
        test_dir=None,
        positive_p=0.66,
        signal_amplifier=1.0,
        match_time=False,
        fillna=False,
        is_test=False,
        shift_range=(-150, 150),
        rotate_range=(0, 0),
        amp_range=(1.0, 1.0),
        preprocess=None,
        transforms=None,
        return_mask=None,
        random_state=0,
        cache_limit=0, # in GB
        ):
        self.df = df
        self.data_dir = data_dir
        with open(test_stat, 'rb') as f:
            self.test_stat = pickle.load(f)
        self.test_dir = test_dir
        self.positive_p = positive_p
        self.match = match_time
        self.fillna = fillna
        self.shift_range = shift_range
        self.rotate_range = rotate_range
        self.amp_range = amp_range
        self.preprocess = preprocess
        self.transforms = transforms
        self.is_test = is_test
        self.cache = {'size': 0}
        self.cache_limit = cache_limit
        self.return_mask = return_mask
        if isinstance(signal_amplifier, (int, float)):
            self.signal_amp = [signal_amplifier]
        else:
            self.signal_amp = signal_amplifier
        self._epoch = 0
        np.random.RandomState(random_state)
        np.random.seed(random_state)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self._load_noise_and_signal(index)

    def _load_real_noise(self, r, return_sft=False):
        if r['id'] in self.cache.keys() and self.is_test:
            spec_h1, spec_l1, time_h1, time_l1 = self.cache[r['id']]
        else:
            if self.is_test:
                if 'path' in r.index:
                    fname = r['path']
                else:
                    fname = self.data_dir/f'{r.id}.pickle'
            else: # in training always read test file
                fname = self.test_dir/f'{r.base_id}.pickle'

            with open(fname, 'rb') as f:
                data = pickle.load(f)

            if 'H1' in data.keys(): # simple format
                spec_h1, time_h1 = data['H1']['spectrogram'].astype(np.float32), data['H1']['timestamps']
                spec_l1, time_l1 = data['L1']['spectrogram'].astype(np.float32), data['L1']['timestamps']
            else: # original format
                gid = list(data.keys())[0]
                data = data[gid]
                spec_h1, time_h1 = data['H1']['SFTs']*1e22, data['H1']['timestamps_GPS']
                spec_l1, time_l1 = data['L1']['SFTs']*1e22, data['L1']['timestamps_GPS']
                if not return_sft:
                    spec_h1 = spec_h1.real**2 + spec_h1.imag**2
                    spec_l1 = spec_l1.real**2 + spec_l1.imag**2

            if self.cache['size'] < self.cache_limit and self.is_test:
                self.cache[r['id']] = (spec_h1, spec_l1, time_h1, time_l1)
                self.cache['size'] += (spec_h1.nbytes + spec_l1.nbytes) / (1024 ** 3)
            else:
                pass

        return spec_h1, spec_l1, time_h1, time_l1

    def _genereta_noise(self, r):
        test_stat = self.test_stat[r['base_id']]
        asd_time_h1, asd_freq_h1, time_h1 = test_stat['H1']
        asd_time_l1, asd_freq_l1, time_l1 = test_stat['L1']
        if asd_time_h1.max() < 5 and asd_freq_h1.max() < 5:
            sft_gen_h1 = reconstruct_from_stat_complex(asd_time_h1.clip(0, 5))
        else:
            sft_gen_h1 = reconstruct_from_stat_complex(asd_time_h1.clip(0, 5))
            sft_org = self._load_real_noise(r, return_sft=True)[0]
            artifact_map = extract_artifact(sft_org, ARTIFACT_NSIGMA)
            sft_gen_h1[np.where(artifact_map)] = sft_org[np.where(artifact_map)]

        if asd_time_l1.max() < 5 and asd_freq_l1.max() < 5:
            sft_gen_l1 = reconstruct_from_stat_complex(asd_time_l1.clip(0, 5))
        else:
            sft_gen_l1 = reconstruct_from_stat_complex(asd_time_l1.clip(0, 5))
            sft_org = self._load_real_noise(r, return_sft=True)[1]
            artifact_map = extract_artifact(sft_org, ARTIFACT_NSIGMA)
            sft_gen_l1[np.where(artifact_map)] = sft_org[np.where(artifact_map)]

        return sft_gen_h1, sft_gen_l1, time_h1, time_l1

    def _preprocess_data(self, r):
        if self.is_test:
            spec_h1, spec_l1, time_h1, time_l1 = self._load_real_noise(r, return_sft=False)
            output_dtype = np.float32
        else:
            spec_h1, spec_l1, time_h1, time_l1 = self._genereta_noise(r)
            output_dtype = np.complex64

        ref_time = min(time_h1.min(), time_l1.min())
        frame_h1 = ((time_h1 - ref_time) / 1800).round().astype(np.uint64)
        frame_l1 = ((time_l1 - ref_time) / 1800).round().astype(np.uint64)

        if self.match:
            _spec = np.full((2, 360, 5760), 0., output_dtype)
            _spec[0][:, frame_h1[frame_h1 < 5760]] = spec_h1[:, frame_h1 < 5760]
            _spec[1][:, frame_l1[frame_l1 < 5760]] = spec_l1[:, frame_l1 < 5760]
            spec_h1, spec_l1 = _spec[0], _spec[1]
        else:
            if spec_h1.shape[1] < spec_l1.shape[1]:
                frame_l1 = frame_l1[:spec_h1.shape[1]]
                spec_l1 = spec_l1[:, :spec_h1.shape[1]]
            elif spec_h1.shape[1] > spec_l1.shape[1]:
                frame_h1 = frame_h1[:spec_l1.shape[1]]
                spec_h1 = spec_h1[:, :spec_l1.shape[1]]

        img = np.stack((spec_h1, spec_l1), axis=2) # (360, t, 2)

        return img, frame_h1, frame_l1

    def _load_signal(self, r):
        if r['id'] in self.cache.keys():
            sft_s = self.cache[r['id']]
        else:
            if 'path' in r.index:
                fname = r['path']
            else:
                fname = self.data_dir/f'{r.id}.pickle'
            with open(fname, 'rb') as f:
                data = pickle.load(f)
            sft_s, _ = data['sft']*1e22, data['timestamps']

            if self.cache['size'] < self.cache_limit:
                self.cache[r['id']] = sft_s
                self.cache['size'] += sft_s.nbytes / (1024 ** 3)
            else:
                pass
        
        shift_y = np.random.randint(*self.shift_range)
        sft_s = np.roll(sft_s, shift_y, axis=0)
        if shift_y > 0:
            sft_s[:shift_y, :] = 0
        else:
            sft_s[shift_y:, :] = 0
        if self.rotate_range[0] != 0 or self.rotate_range[0] != 0:
            rotate_ang = np.random.randint(*self.rotate_range)
            sft_s = ndimage.rotate(sft_s, rotate_ang, reshape=False)
        if self.amp_range[0] != 1 or self.amp_range[0] != 1:
            amp = np.random.uniform(*self.amp_range)
            sft_s *= amp
        sft_s *= self.signal_amp[min(self._epoch, len(self.signal_amp)-1)]
        return sft_s

    def _inject_signal(self, noise, signal, frame_h1, frame_l1):
        # align timestamps
        if self.match:
            noise[:, frame_h1[frame_h1 < 5760], 0] += signal[:360, frame_h1[frame_h1 < 5760]]
            noise[:, frame_l1[frame_l1 < 5760], 1] += signal[:360, frame_l1[frame_l1 < 5760]]
            signal_mask = np.zeros((noise.shape[0], noise.shape[1]), dtype=np.float32)
            if self.return_mask:
                signal = signal.real**2 + signal.imag**2
                signal_bin = ((signal - signal.min()) / (signal.max() - signal.min() + 1e-6) > 0.25).astype(np.float32)   
                signal_mask[:, frame_h1[frame_h1 < 5760]] = signal_bin[:360, frame_h1[frame_h1 < 5760]]
                signal_mask[:, frame_l1[frame_l1 < 5760]] = signal_bin[:360, frame_l1[frame_l1 < 5760]]
        else:
            noise[:, :, 0] += signal[:360, frame_h1]
            noise[:, :, 1] += signal[:360, frame_l1]
            signal_mask = np.zeros((noise.shape[0], noise.shape[1]), dtype=np.float32)
            if self.return_mask:
                signal = signal.real**2 + signal.imag**2
                signal_bin = ((signal - signal.min()) / (signal.max() - signal.min() + 1e-6) > 0.25).astype(np.float32)   
                signal_mask[:, :] = signal_bin[:, frame_h1]
                signal_mask[:, :] = signal_bin[:, frame_l1]
        signal_mask[signal_mask!=signal_mask] = 0
        return noise, signal_mask[:, :, None]

    def _load_noise_and_signal(self, index):
        if self.is_test: # test
            r = self.df.iloc[index]
            img, _, _ = self._preprocess_data(r)
            signal_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
            target = torch.tensor([r['target']]).float()

        else: # train
            if np.random.random() < self.positive_p:
                signal_r = self.df.iloc[index]
                signal_spec = self._load_signal(signal_r)
                noise_spec, frame_h1, frame_l1 = self._preprocess_data(signal_r)

                img, signal_mask = self._inject_signal(noise_spec, signal_spec, frame_h1, frame_l1)
                target = torch.tensor([1]).float()
            else:
                signal_r = self.df.iloc[index]
                noise_spec, frame_h1, frame_l1 = self._preprocess_data(signal_r)

                img = noise_spec
                signal_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
                target = torch.tensor([0]).float()
            img = (img.real**2 + img.imag**2).astype(np.float32)

        if self.preprocess:
            if self.return_mask:
                t = self.preprocess(image=img, mask=signal_mask)
                img = t['image']
                signal_mask = t['mask']
            else:
                t = self.preprocess(image=img)
                img = t['image']
        
        if self.transforms:
            if self.return_mask:
                t = self.transforms(image=img, mask=signal_mask)
                img = t['image']
                signal_mask = t['mask']
            else:
                t = self.transforms(image=img)
                img = t['image']

        if self.return_mask:
            return img, signal_mask, target
        else:
            return img, target

    def step(self):
        self._epoch += 1