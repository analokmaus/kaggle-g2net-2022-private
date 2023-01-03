# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import scipy.signal
import pandas as pd
from tqdm.auto import tqdm
import gc
import pickle

# %%
PATH = 'input/g2net-detecting-continuous-gravitational-waves/test'
PATH_EX = 'input/g2net-detecting-continuous-gravitational-waves/external'
OUT = 'input/g2net-detecting-continuous-gravitational-waves/denoise_H.pickle'

# files = [os.path.join(PATH_EX,f) for f in sorted(os.listdir(PATH_EX))]
files = [str(path) for path in Path(PATH_EX).glob('*.hdf5')]
T = 1800
# SR = 16384 # !!! change if work with 4096 SR data
SR = 4096
SZ = 360
TH = 1.5
SOURCE = 'H1_SFTs_amplitudes'


df = pd.read_csv('input/test_stationery.csv')
df = df.loc[~df.stationery]
df.head()

# %%
def read_hdf5(fname):
    with h5py.File(fname, 'r') as f:
        strain = f['strain']['Strain'][:]
        ts = f['strain']['Strain'].attrs['Xspacing']

        metaKeys = f['meta'].keys()
        meta = f['meta']
        gpsStart = meta['GPSstart'][()]
        duration = meta['Duration'][()]
        has_nan = strain[np.isnan(strain)].size > 0
    return {'strain':strain, 'ts':ts, 
            'gpsStart':gpsStart, 'duration':duration, 'has_nan':has_nan}


def extract_data_from_hdf5(path):
    data = {}
    # with h5py.File(path, "r") as f:
    #     ID_key = list(f.keys())[0]
    #     # Retrieve the frequency data
    #     data['freq'] = np.array(f[ID_key]['frequency_Hz'])
    #     # Retrieve the Livingston decector data
    #     data['L1_SFTs_amplitudes'] = np.array(f[ID_key]['L1']['SFTs'])
    #     data['L1_ts'] = np.array(f[ID_key]['L1']['timestamps_GPS'])
    #     # Retrieve the Hanford decector data
    #     data['H1_SFTs_amplitudes'] = np.array(f[ID_key]['H1']['SFTs'])
    #     data['H1_ts'] = np.array(f[ID_key]['H1']['timestamps_GPS'])
    with open(path, 'rb') as fp:
        f = pickle.load(fp)
        ID_key = list(f.keys())[0]
        # Retrieve the frequency data
        data['freq'] = np.array(f[ID_key]['frequency_Hz'])
        # Retrieve the Livingston decector data
        data['L1_SFTs_amplitudes'] = np.array(f[ID_key]['L1']['SFTs'])
        data['L1_ts'] = np.array(f[ID_key]['L1']['timestamps_GPS'])
        # Retrieve the Hanford decector data
        data['H1_SFTs_amplitudes'] = np.array(f[ID_key]['H1']['SFTs'])
        data['H1_ts'] = np.array(f[ID_key]['H1']['timestamps_GPS'])
    return data


class Model_FFT(nn.Module):
    def __init__(self, N=SR*T, sr=SR):
        super().__init__()
        window = scipy.signal.windows.tukey(N, 0.001)
        self.window = nn.Parameter(torch.from_numpy(window),requires_grad=False)
        self.range = [89500,901500] #50-500 Hz
        self.freq = (np.fft.rfftfreq(N)*sr)[self.range[0]:self.range[1]]
        self.sr, self.N = sr, N
        
    def forward(self, x):
        with torch.no_grad():
            ys,shifts = [],[]
            for i in range(0,x.shape[-1] - self.N, self.sr):
                xi = x[i:i+self.N]
                if torch.isnan(xi).any(-1): continue
                y = torch.fft.rfft(xi*self.window)[self.range[0]:self.range[1]] / self.sr
                y = (y*1e22).abs().float().cpu()
                ys.append(y)
                shifts.append(i//self.sr)
        return torch.stack(ys,0), torch.LongTensor(shifts)

# %%
fft_model = Model_FFT().cuda()
freq = fft_model.freq

denoised_data = {}
for index, row in tqdm(df.iterrows(), total=len(df)):
    idx = row['id']
    # data_src = extract_data_from_hdf5(os.path.join(PATH, idx+'.hdf5'))
    data_src = extract_data_from_hdf5(os.path.join(PATH, idx+'.pickle'))
    src = torch.zeros(data_src[SOURCE].shape)
    denoised_data[idx] = src

# %%
for fname in tqdm(files):
    match_num = 0
    data = torch.from_numpy(read_hdf5(fname)['strain'])
    try:
        stfts, shifts = fft_model(data.float().cuda())
    except:
        print(f'error: {fname}')
        del data; gc.collect()
        continue
    del data
    
    for index, row in df.iterrows():
        idx = row['id']
        # data_src = extract_data_from_hdf5(os.path.join(PATH, idx+'.hdf5'))
        data_src = extract_data_from_hdf5(os.path.join(PATH, idx+'.pickle'))
        freq_start = (np.abs(freq - data_src['freq'][0])).argmin()

        tgt = stfts[:,freq_start:freq_start+SZ]
        src = torch.from_numpy(np.abs(data_src[SOURCE]*1e22)).permute(1,0)
        dists = torch.cdist(src.clip(0, 15).cuda(),tgt.clip(0, 15).cuda()).cpu()

        if dists.min() < TH:
            values,indices = dists.min(-1)
            #print(indices[values < TH], torch.where(values < TH)) ##
            denoised_data[idx][:,values < TH] = (src[values < TH] - tgt[indices[values < TH]]).T
            match_num += 1
    print(fname, match_num)
    gc.collect()
    #break ##

# %%
with open(OUT, 'wb') as f:
    pickle.dump(denoised_data, f, protocol=pickle.HIGHEST_PROTOCOL)
