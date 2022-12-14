import pickle
from prep_data_v18v import to_spectrogram
from transforms import adaptive_resize
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool


def get_metadata(gid):
    with open(f'input/g2net-detecting-continuous-gravitational-waves/v18v/{gid}.pickle', 'rb') as f:
        data = pickle.load(f)[gid]
    freq = data['frequency_Hz'].mean()
    spec_h1 = adaptive_resize(to_spectrogram(data['H1']['SFTs']*1e22)[:, :, None], 360, np.mean)
    ts_h1 = data['H1']['timestamps_GPS']
    spec_l1 = adaptive_resize(to_spectrogram(data['L1']['SFTs']*1e22)[:, :, None], 360, np.mean)
    ts_l1 = data['L1']['timestamps_GPS']
    asd_h1 = spec_h1.mean(axis=(1, 2)).reshape(180, 2).mean(1)
    std_h1 = spec_h1.std()
    min_h1 = spec_h1.min()
    peak_h1 = np.max((spec_h1 - min_h1) / std_h1)
    asd_l1 = spec_l1.mean(axis=(1, 2)).reshape(180, 2).mean(1)
    std_l1 = spec_l1.std()
    min_l1 = spec_l1.min()
    peak_l1 = np.max((spec_l1 - min_l1) / std_h1)
    return {
        'id': gid, 
        'freq': freq, 
        'nonstationary': int(np.percentile(asd_h1, 90)-np.percentile(asd_h1, 10) / asd_h1.mean() > 1.5),
        'artifact': int(max(peak_h1, peak_l1) > 25)}


valid = pd.read_csv('input/g2net-detecting-continuous-gravitational-waves/v18v.csv')
with Pool(40) as p:
    metadata = p.map(
        get_metadata, 
        valid['id'].values)
metadata = pd.DataFrame(metadata)

valid.merge(metadata, on='id', how='left').to_csv(
    'input/g2net-detecting-continuous-gravitational-waves/v18v.csv', index=False)