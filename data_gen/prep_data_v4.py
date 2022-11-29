# import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import pyfstat
from pyfstat.utils import get_sft_as_arrays
import math
import random
from multiprocessing import Pool
from functools import partial
import pickle


DATASET = 'v4'
NUM_POSITIVE = 6500
NUM_NEGATIVE = 3500
NUM_WORKERS = 8
EXPORT_DIR = Path(f'input/g2net-detecting-continuous-gravitational-waves/{DATASET}/')
EXPORT_DIR.mkdir(exist_ok=True)



def make_signal(idx, test_timestamps):
    signal_depth = np.random.uniform(5, 50)
    signal_center = np.random.uniform(50, 500)
    timestamps_set = random.choice(test_timestamps)
    timestamps_h1, timestamps_l1 = timestamps_set['H1'], timestamps_set['L1']
    sqrt_ratio = np.random.uniform(0.75, 1.25)
    noise_kwargs_h1 = {
        "outdir": 'pyfstat',
        "label": f'signal{idx}_h1',
        "timestamps": timestamps_h1,
        # "duration": 4 * 30 * 86400,
        # "Tsft": 1800,
        "detectors": "H1",
        "sqrtSX": np.random.uniform(1.5e-23, 7.5e-24),
        "SFTWindowType": "tukey",
        "SFTWindowBeta": 0.01,
        "Band": 0.4,
    }
    noise_kwargs_l1 = noise_kwargs_h1.copy()
    noise_kwargs_l1['label'] = f'signal{idx}_l1'
    noise_kwargs_l1['detectors'] = 'L1'
    noise_kwargs_l1['timestamps'] = timestamps_l1
    noise_kwargs_l1['sqrtSX'] = noise_kwargs_h1['sqrtSX'] * sqrt_ratio
   
    signal_kwargs_h1 = {
        "F0": signal_center,
        # "F1": -1 * (10**np.random.uniform(-12, -8)),
        "F1": np.random.uniform(-1e-12, 1e-12),
        "F2": 0,
        "Alpha": np.random.uniform(0, math.pi * 2),
        "Delta": np.random.uniform(-math.pi/2, math.pi/2),
        "h0": noise_kwargs_h1['sqrtSX'] / signal_depth,
        "cosi": np.random.uniform(-1, 1),
        "psi": np.random.uniform(-math.pi/4, math.pi/4),
        "phi": np.random.uniform(0, math.pi*2),
        "tref": timestamps_h1[0],
    }
    signal_kwargs_l1 = signal_kwargs_h1.copy()
    signal_kwargs_l1['h0'] = noise_kwargs_l1['sqrtSX'] / signal_depth
    signal_kwargs_l1['tref'] = timestamps_l1[0]

    writer_h1 = pyfstat.Writer(**noise_kwargs_h1, **signal_kwargs_h1)
    writer_h1.make_data()
    writer_l1 = pyfstat.Writer(**noise_kwargs_l1, **signal_kwargs_l1)
    writer_l1.make_data()
    freqs_h1, times_h1, sft_data_h1 = get_sft_as_arrays(writer_h1.sftfilepath)
    freqs_l1, times_l1, sft_data_l1 = get_sft_as_arrays(writer_l1.sftfilepath)
    freqs = freqs_h1
    times = {'H1': times_h1['H1'], 'L1': times_l1['L1']}
    sft_data = {'H1': sft_data_h1['H1'], 'L1': sft_data_l1['L1']}
    slice_start = np.random.randint(75, 285)
    sft_crop = {}
    for d in ['H1', 'L1']:
        sft_crop[d] = sft_data[d][slice_start:slice_start+360]
    freqs = freqs[slice_start:slice_start+360]
    for fname in [
        f'{noise_kwargs_h1["label"]}.cff',
        f'{noise_kwargs_l1["label"]}.cff',
        f'H-{len(timestamps_h1)}_H1_1800SFT_{noise_kwargs_h1["label"]}-{timestamps_h1[0]}-{timestamps_h1[-1]-timestamps_h1[0]+1800}.sft',
        f'L-{len(timestamps_l1)}_L1_1800SFT_{noise_kwargs_l1["label"]}-{timestamps_l1[0]}-{timestamps_l1[-1]-timestamps_l1[0]+1800}.sft',
        f'{noise_kwargs_h1["label"]}_timestamps_H1.csv',
        f'{noise_kwargs_l1["label"]}_timestamps_L1.csv']:
        (Path(f'{noise_kwargs_h1["outdir"]}')/fname).unlink()
    instance_id = f'pos_{idx}'
    return {instance_id: {
            'H1': {
                'SFTs': sft_crop['H1'], 
                'timestamps_GPS': times['H1']
            },
            'L1': {
                'SFTs': sft_crop['L1'], 
                'timestamps_GPS': times['L1']
            },
            'frequency_Hz': freqs}}, {
            'H1': dict(noise_kwargs_h1, **signal_kwargs_h1),
            'L1': dict(noise_kwargs_l1, **signal_kwargs_l1),
            'signal_depth': signal_depth
        }


def make_noise(idx, test_timestamps):
    signal_center = np.random.uniform(50, 500)
    timestamps_set = random.choice(test_timestamps)
    timestamps_h1, timestamps_l1 = timestamps_set['H1'], timestamps_set['L1']
    sqrt_ratio = np.random.uniform(0.75, 1.25)
    noise_kwargs_h1 = {
        "outdir": 'pyfstat',
        "label": f'signal{idx}_h1',
        "timestamps": timestamps_h1,
        # "duration": 4 * 30 * 86400,
        # "Tsft": 1800,
        "detectors": "H1",
        "sqrtSX": np.random.uniform(1.5e-23, 7.5e-24),
        "SFTWindowType": "tukey",
        "SFTWindowBeta": 0.01,
        "F0": signal_center,
        "Band": 0.4,
    }
    noise_kwargs_l1 = noise_kwargs_h1.copy()
    noise_kwargs_l1['label'] = f'signal{idx}_l1'
    noise_kwargs_l1['detectors'] = 'L1'
    noise_kwargs_l1['timestamps'] = timestamps_l1
    noise_kwargs_l1['sqrtSX'] = noise_kwargs_h1['sqrtSX'] * sqrt_ratio

    writer_h1 = pyfstat.Writer(**noise_kwargs_h1)
    writer_h1.make_data()
    writer_l1 = pyfstat.Writer(**noise_kwargs_l1)
    writer_l1.make_data()
    freqs_h1, times_h1, sft_data_h1 = get_sft_as_arrays(writer_h1.sftfilepath)
    freqs_l1, times_l1, sft_data_l1 = get_sft_as_arrays(writer_l1.sftfilepath)
    freqs = freqs_h1
    times = {'H1': times_h1['H1'], 'L1': times_l1['L1']}
    sft_data = {'H1': sft_data_h1['H1'], 'L1': sft_data_l1['L1']}
    slice_start = np.random.randint(75, 285)
    sft_crop = {}
    for d in ['H1', 'L1']:
        sft_crop[d] = sft_data[d][slice_start:slice_start+360]
    freqs = freqs[slice_start:slice_start+360]
    for fname in [
        f'H-{len(timestamps_h1)}_H1_1800SFT_{noise_kwargs_h1["label"]}-{timestamps_h1[0]}-{timestamps_h1[-1]-timestamps_h1[0]+1800}.sft',
        f'L-{len(timestamps_l1)}_L1_1800SFT_{noise_kwargs_l1["label"]}-{timestamps_l1[0]}-{timestamps_l1[-1]-timestamps_l1[0]+1800}.sft',
        f'{noise_kwargs_h1["label"]}_timestamps_H1.csv',
        f'{noise_kwargs_l1["label"]}_timestamps_L1.csv']:
        (Path(f'{noise_kwargs_h1["outdir"]}')/fname).unlink()
    instance_id = f'neg_{idx}'
    return {instance_id: {
            'H1': {
                'SFTs': sft_crop['H1'], 
                'timestamps_GPS': times['H1']
            },
            'L1': {
                'SFTs': sft_crop['L1'], 
                'timestamps_GPS': times['L1']
            },
            'frequency_Hz': freqs}}, {
            'H1': noise_kwargs_h1,
            'L1': noise_kwargs_l1,
            'signal_depth': 1000
        }


def generate_sample(index, test_timestamps, positive=False):
    if positive:
        results, metadata = make_signal(index, test_timestamps)
        target = 1
    else:
        results, metadata = make_noise(index, test_timestamps)
        target = 0
    instance_id = list(results.keys())[0]
    with open(EXPORT_DIR/f'{instance_id}.pickle', 'wb') as f:
        pickle.dump(results, f)
    # save_dict_to_hdf5(results, EXPORT_DIR/f'{instance_id}.hdf5')
    metadata2 = {}
    for d in ['H1', 'L1']:
        for k, v in metadata[d].items():
            if k in ['label', 'timestamps']:
                continue
            metadata2[f'{d}_{k}'] = v
    metadata2['id'] = instance_id
    metadata2['signal_depth'] = metadata['signal_depth']
    metadata2['target'] = target
    return metadata2


if __name__ == '__main__':
    # get gaps
    test = pd.read_csv('input/g2net-detecting-continuous-gravitational-waves/sample_submission.csv')
    ts_with_gaps = []
    for i, gid in enumerate(test['id'].values):
        fname = Path('input/g2net-detecting-continuous-gravitational-waves/test')/f'{gid}.pickle'
        with open(fname, 'rb') as fp:
            f = pickle.load(fp)
            freq = list(f[gid]['frequency_Hz'])
            sig_h1, time_h1 = f[gid]['H1']['SFTs'], np.array(f[gid]['H1']['timestamps_GPS'])
            sig_l1, time_l1 = f[gid]['L1']['SFTs'], np.array(f[gid]['L1']['timestamps_GPS'])
            ts_with_gaps.append({'H1': time_h1, 'L1': time_l1})
        if i % 100 == 0:
            print(f'{i} done.')
    with open('input/g2net-detecting-continuous-gravitational-waves/test_gaps.pickle', 'wb') as f:
        pickle.dump(ts_with_gaps, f)
    # generate samples
    with Pool(NUM_WORKERS) as p:
        metadata_pos = p.map(
            partial(generate_sample, test_timestamps=ts_with_gaps, positive=True), 
            range(NUM_POSITIVE))
    with Pool(NUM_WORKERS) as p:
        metadata_neg = p.map(
            partial(generate_sample, test_timestamps=ts_with_gaps, positive=False), 
            range(NUM_NEGATIVE))
    metadata = metadata_pos + metadata_neg
    pd.DataFrame(metadata).to_csv(f'input/g2net-detecting-continuous-gravitational-waves/{DATASET}.csv', index=False)
