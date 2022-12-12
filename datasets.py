import numpy as np
import os
import torch
import torch.utils.data as D
import matplotlib.pyplot as plt
# import h5py
import pickle
from pathlib import Path
from scipy.stats import norm


'''

'''
def laeyoung_normalize(sft):
        sft = np.abs(sft) ** 2
        pos = int(sft.size * 0.99903)
        exp = norm.ppf((pos + 0.4) / (sft.size + 0.215))
        scale = np.partition(sft.flatten(), pos, -1)[pos]
        sft /= (scale / exp.astype(scale.dtype) ** 2)
        return sft


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
                gid = list(data.keys())[0]
                data = data[gid]
            spec_h1, time_h1 = data['H1']['SFTs']*1e22, data['H1']['timestamps_GPS']
            spec_l1, time_l1 = data['L1']['SFTs']*1e22, data['L1']['timestamps_GPS']

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
