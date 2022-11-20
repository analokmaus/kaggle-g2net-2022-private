import numpy as np
from multiprocessing import Pool
import torch
import torch.utils.data as D
import matplotlib.pyplot as plt
# import h5py
import pickle
from pathlib import Path


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
        spec_diff=False,
        resize_factor=1,
        transforms=None):
        self.df = df
        self.data_dir = data_dir
        self.norm = normalize
        self.match = match_time
        self.diff = spec_diff
        self.resize_f = resize_factor
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img, target = self._load_spec(index)
        return img, target

    def _match_time(self, spec1, spec2, time1, time2):
        assert len(time1) == spec1.shape[1]
        assert len(time2) == spec2.shape[1]
        if len(time1) < len(time2):
            match_idx = np.searchsorted(time2, time1) - 1
            return spec1, spec2[:, match_idx]
        else:
            match_idx = np.searchsorted(time1, time2) - 1
            return spec1[:, match_idx], spec2

    def _load_spec(self, index):
        r = self.df.iloc[index]
        target = torch.tensor([r['target']]).float()
        img = np.empty((2, 360, 4096//self.resize_f), dtype=np.float32)
        fname = self.data_dir/f'{r.id}.pickle'
        with open(fname, 'rb') as f:
            data = pickle.load(f)[r['id']]
        spec_h1, time_h1 = data['H1']['SFTs']*1e22, data['H1']['timestamps_GPS']
        spec_l1, time_l1 = data['L1']['SFTs']*1e22, data['L1']['timestamps_GPS']
        if self.match:
            spec_h1, spec_l1 = self._match_time(spec_h1, spec_l1, time_h1, time_l1)
        else:
            if spec_h1.shape[1] < spec_l1.shape[1]:
                spec_l1 = spec_l1[:, :spec_h1.shape[1]]
            else:
                spec_h1 = spec_h1[:, :spec_l1.shape[1]]
        
        spec_h1 = spec_h1.real**2 + spec_h1.imag**2
        spec_l1 = spec_l1.real**2 + spec_l1.imag**2
        if self.norm == 'local':
            spec_h1 /= np.mean(spec_h1)
            spec_l1 /= np.mean(spec_l1) 
        elif self.norm == 'global':
            spec_h1 /= 13.
            spec_l1 /= 13.
        else:
            pass

        if self.resize_f != 1:
            spec_h1 = spec_h1[:, :4096].reshape(360, 4096//self.resize_f, self.resize_f).mean(2)
            spec_l1 = spec_l1[:, :4096].reshape(360, 4096//self.resize_f, self.resize_f).mean(2)

        if self.diff:
            img = np.stack((spec_h1, spec_l1, spec_h1 - spec_l1), axis=2) # (360, t, 3)
        else:
            img = np.stack((spec_h1, spec_l1), axis=2) # (360, t, 2)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        return img, target
    