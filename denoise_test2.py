import numpy as np
import pandas as pd
import os,gc,random
import pickle
from tqdm.auto import tqdm
from collections import OrderedDict
import h5py
import torch
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


PATH = 'input/g2net-detecting-continuous-gravitational-waves/test/'
OUT = 'input/g2net-detecting-continuous-gravitational-waves/leak_2.pickle'

df = pd.read_csv('input/test_stationery.csv')
df = df.loc[~df.stationery]
print(df.head())


def extract_data_from_hdf5(path):
    data = {}
    with h5py.File(path, "r") as f:
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


def get_correlation(x1,x2):
    correlation = (torch.einsum('ki,kj->ij', x1, x2)/x1.shape[0] - \
                   torch.einsum('i,j->ij', x1.mean(0), x2.mean(0)))/ \
                   torch.einsum('i,j->ij', x1.std(0), x2.std(0))
    return correlation


clean_data = {}
TH = 0.95
denoised_target = {}
for index, row in tqdm(df.iterrows(), total=len(df)):
    idx, freq_origin = row[['id','freq']]
    data_src = extract_data_from_hdf5(os.path.join(PATH, idx+'.hdf5')) 
    buf = {
        'H1': torch.zeros(data_src['H1_SFTs_amplitudes'].shape),
        'L1': torch.zeros(data_src['L1_SFTs_amplitudes'].shape),
        'H1_mask': torch.zeros(data_src['H1_SFTs_amplitudes'].shape, dtype=torch.uint8),
        'L1_mask': torch.zeros(data_src['L1_SFTs_amplitudes'].shape, dtype=torch.uint8),
        'denoise_pair_H': [],
        'denoise_pair_L': [],
        'coverage_H1': 0,
        'coverage_L1': 0,
        'empty' : True
    }
    for offset in np.arange(-3, 4):
        freq = freq_origin
        freq += offset
        neigbors = df.loc[np.abs(df.freq.values - freq) < 0.2, ['id','freq']].set_index('id').to_dict()['freq']
        if len(neigbors) == 0: continue
        
        for key in neigbors.keys():
            if key == idx:
                continue
            dfreq = round(1800*(freq - neigbors[key]))
            if 360 - abs(dfreq) < 40: continue
            f1s,f2s = max(-dfreq,0), min(len(data_src['freq']) - dfreq,len(data_src['freq']))
            f1t,f2t = max(dfreq,0), min(len(data_src['freq']) + dfreq,len(data_src['freq']))
            f1s,f2s = max(-dfreq,0), min(len(data_src['freq']) - dfreq,len(data_src['freq']))
            f1t,f2t = max(dfreq,0), min(len(data_src['freq']) + dfreq,len(data_src['freq']))
            data_tgt = extract_data_from_hdf5(os.path.join(PATH, key+'.hdf5'))
            
            src_H = torch.from_numpy(np.abs(data_src['H1_SFTs_amplitudes']*1e22))
            tgt_H = torch.from_numpy(np.abs(data_tgt['H1_SFTs_amplitudes']*1e22))
            correlation_H = get_correlation(src_H[f1s:f2s],tgt_H[f1t:f2t])
            src_L = torch.from_numpy(np.abs(data_src['L1_SFTs_amplitudes']*1e22))
            tgt_L = torch.from_numpy(np.abs(data_tgt['L1_SFTs_amplitudes']*1e22))
            correlation_L = get_correlation(src_L[f1s:f2s],tgt_L[f1t:f2t])
            
            if correlation_H.max() > TH:
                values,indices = correlation_H.max(-1)
                dif_abs = (src_H[f1s:f2s, values > TH] - tgt_H[f1t:f2t,indices[values > TH]]).abs()
                min_val = torch.min(buf['H1'][f1s:f2s,values > TH], dif_abs)
                buf['H1'][f1s:f2s,values > TH] = torch.where(buf['H1_mask'][f1s:f2s,values > TH] == 0,
                        dif_abs, min_val)
                buf['H1_mask'][f1s:f2s,values > TH] += 1
                buf['empty'] = False
                buf['denoise_pair_H'].append(key)

            if correlation_L.max() > TH:
                values,indices = correlation_L.max(-1)
                dif_abs = (src_L[f1s:f2s, values > TH] - tgt_L[f1t:f2t,indices[values > TH]]).abs()
                min_val = torch.min(buf['L1'][f1s:f2s,values > TH], dif_abs)
                buf['L1'][f1s:f2s,values > TH] = torch.where(buf['L1_mask'][f1s:f2s,values > TH] == 0,
                        dif_abs, min_val)
                buf['L1_mask'][f1s:f2s,values > TH] += 1
                buf['empty'] = False
                buf['denoise_pair_L'].append(key)

        
    if not buf['empty']:
        del buf['empty']
        buf['coverage_H1'] = (buf['H1_mask'].amax(1) > 0).sum() / 360
        buf['coverage_L1'] = (buf['L1_mask'].amax(1) > 0).sum() / 360
        # buf['signal'] = buf['H1'].max() > 0.01 or buf['L1'].max() > 0.01
        # buf['signal_stats'] = (buf['H1'].sum(), buf['L1'].max())
        clean_data[idx] = buf
        #if buf['H1'].max() > 0.01: break
        #if buf['H1_mask'].max() > 1: break

with open(OUT, 'wb') as handle:
    pickle.dump(clean_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
