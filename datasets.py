import numpy as np
from multiprocessing import Pool
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
class G2Net2022Dataset(D.Dataset):
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
                spec_h1 = spec_h1.reshape(360, 5760//self.resize_f, self.resize_f).mean(2)
                spec_l1 = spec_l1.reshape(360, 5760//self.resize_f, self.resize_f).mean(2)
            else:
                if (not self.is_test) and self.random_crop:
                    slice_start = np.random.randint(0, spec_h1.shape[1] - 4096)
                    spec_h1 = spec_h1[:, slice_start:slice_start+4096].reshape(360, 4096//self.resize_f, self.resize_f).mean(2)
                    spec_l1 = spec_l1[:, slice_start:slice_start+4096].reshape(360, 4096//self.resize_f, self.resize_f).mean(2)
                else:
                    img_size = spec_h1.shape[1]
                    img_size2 = int((img_size // self.resize_f) * self.resize_f)
                    spec_h1 = spec_h1[:, :img_size2].reshape(360, img_size2//self.resize_f, self.resize_f).mean(2)
                    spec_l1 = spec_l1[:, :img_size2].reshape(360, img_size2//self.resize_f, self.resize_f).mean(2)
                
        if self.diff:
            img = np.stack((spec_h1, spec_l1, spec_h1 - spec_l1), axis=2) # (360, t, 3)
        else:
            img = np.stack((spec_h1, spec_l1), axis=2) # (360, t, 2)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        return img, target


class G2Net2022Dataset2(D.Dataset):
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
                spec_h1 = spec_h1.reshape(360, 5760//self.resize_f, self.resize_f).mean(2)
                spec_l1 = spec_l1.reshape(360, 5760//self.resize_f, self.resize_f).mean(2)
            else:
                if (not self.is_test) and self.random_crop:
                    slice_start = np.random.randint(0, spec_h1.shape[1] - 4096)
                    spec_h1 = spec_h1[:, slice_start:slice_start+4096].reshape(360, 4096//self.resize_f, self.resize_f).mean(2)
                    spec_l1 = spec_l1[:, slice_start:slice_start+4096].reshape(360, 4096//self.resize_f, self.resize_f).mean(2)
                else:
                    img_size = spec_h1.shape[1]
                    img_size2 = int((img_size // self.resize_f) * self.resize_f)
                    spec_h1 = spec_h1[:, :img_size2].reshape(360, img_size2//self.resize_f, self.resize_f).mean(2)
                    spec_l1 = spec_l1[:, :img_size2].reshape(360, img_size2//self.resize_f, self.resize_f).mean(2)
                
        if self.diff:
            img = np.stack((spec_h1, spec_l1, spec_h1 - spec_l1), axis=2) # (360, t, 3)
        else:
            img = np.stack((spec_h1, spec_l1), axis=2) # (360, t, 2)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        return img, target
