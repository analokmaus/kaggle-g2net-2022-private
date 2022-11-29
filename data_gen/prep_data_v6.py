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


DATASET = 'v6'
NUM_POSITIVE = 6500
NUM_NEGATIVE = 3500
NUM_WORKERS = 16
SEGMENT_MAX = 12
SX_MIN, SX_MAX = 9e-24, 1.2e-23
F1_MIN, F1_MAX = -1e-8, 1e-8
EXPORT_DIR = Path(f'input/g2net-detecting-continuous-gravitational-waves/{DATASET}/')
EXPORT_DIR.mkdir(exist_ok=True)


def make_segment(timestamps, num_segment):
    if num_segment == 1:
        return [0, len(timestamps)]
    else:
        segment_idx = np.sort(np.random.choice(np.arange(50, len(timestamps)-50), num_segment-1, replace=False))
        return [0] + segment_idx.tolist() + [len(timestamps)]


def make_signal(idx, test_timestamps):
    signal_depth = np.random.uniform(5, 50)
    signal_center = np.random.uniform(50, 500)
    timestamps_set = random.choice(test_timestamps)
    timestamps_h1, timestamps_l1 = timestamps_set['H1'], timestamps_set['L1']
    num_segment = np.random.randint(1, SEGMENT_MAX, size=2)
    segment_indice_h1 = make_segment(timestamps_h1, num_segment[0])
    segment_indice_l1 = make_segment(timestamps_l1, num_segment[1])

    sft_path_h1 = []
    noise_kwargs_h1 = {
        "outdir": 'pyfstat',
        "detectors": "H1",
        "SFTWindowType": "tukey",
        "SFTWindowBeta": 0.01,
        "Band": 0.4,
        "F0": signal_center,
    }
    signal_kwargs_h1 = {
        "outdir": 'pyfstat',
        "label": f'signal{idx}_h1',
        "F0": signal_center,
        "F1": np.random.uniform(F1_MIN, F1_MAX),
        "F2": 0,
        "Alpha": np.random.uniform(0, math.pi * 2),
        "Delta": np.random.uniform(-math.pi/2, math.pi/2),
        "cosi": np.random.uniform(-1, 1),
        "psi": np.random.uniform(-math.pi/4, math.pi/4),
        "phi": np.random.uniform(0, math.pi*2),
        "SFTWindowType": "tukey",
    }
    sqrtSXs_h1 = []
    delete_queue = []
    for seg_i, (seg_sta, seg_end) in enumerate(
        zip(segment_indice_h1[:-1], segment_indice_h1[1:])):
        noise_kwargs_seg = noise_kwargs_h1.copy()
        noise_kwargs_seg['label'] = f'signal{idx}_h1_{seg_i}'
        noise_kwargs_seg['timestamps'] = timestamps_h1[seg_sta:seg_end]
        noise_kwargs_seg['sqrtSX'] = np.random.uniform(SX_MIN, SX_MAX)
        sqrtSXs_h1.append(noise_kwargs_seg['sqrtSX'])
        writer_h1 = pyfstat.Writer(**noise_kwargs_seg)
        writer_h1.make_data()
        sft_path_h1.append(writer_h1.sftfilepath)
        delete_queue.append(
            Path(writer_h1.sftfilepath))
        delete_queue.append(
            Path(f'{noise_kwargs_seg["outdir"]}/{noise_kwargs_seg["label"]}.cff'))
        delete_queue.append(
            Path(f'{noise_kwargs_seg["outdir"]}/{noise_kwargs_seg["label"]}_timestamps_H1.csv'))
    signal_kwargs_h1['noiseSFTs'] = ";".join(sft_path_h1)
    signal_kwargs_h1['h0'] = np.min(sqrtSXs_h1) / signal_depth
    writer_h1 = pyfstat.Writer(**signal_kwargs_h1)
    delete_queue.append(
        Path(f'{signal_kwargs_h1["outdir"]}/{signal_kwargs_h1["label"]}.cff'))
    delete_queue.append(
        Path(f'{signal_kwargs_h1["outdir"]}/{signal_kwargs_h1["label"]}_timestamps_H1.csv'))
    writer_h1.make_data()
    delete_queue.append(
        Path(writer_h1.sftfilepath))

    freqs_h1, times_h1, sft_data_h1 = get_sft_as_arrays(writer_h1.sftfilepath)

    sft_path_l1 = []
    noise_kwargs_l1 = noise_kwargs_h1.copy()
    noise_kwargs_l1['detectors'] = 'L1'
    sqrtSXs_l1 = []
    for seg_i, (seg_sta, seg_end) in enumerate(
        zip(segment_indice_l1[:-1], segment_indice_l1[1:])):
        noise_kwargs_seg = noise_kwargs_l1.copy()
        noise_kwargs_seg['label'] = f'signal{idx}_l1_{seg_i}'
        noise_kwargs_seg['timestamps'] = timestamps_l1[seg_sta:seg_end]
        noise_kwargs_seg['sqrtSX'] = np.random.uniform(SX_MIN, SX_MAX)
        sqrtSXs_l1.append(noise_kwargs_seg['sqrtSX'])
        writer_h1 = pyfstat.Writer(**noise_kwargs_seg)
        writer_h1.make_data()
        writer_h1 = pyfstat.Writer(**noise_kwargs_seg)
        writer_h1.make_data()
        sft_path_l1.append(writer_h1.sftfilepath)
        delete_queue.append(
            Path(writer_h1.sftfilepath))
        delete_queue.append(
            Path(f'{noise_kwargs_seg["outdir"]}/{noise_kwargs_seg["label"]}.cff'))
        delete_queue.append(
            Path(f'{noise_kwargs_seg["outdir"]}/{noise_kwargs_seg["label"]}_timestamps_L1.csv'))
    signal_kwargs_l1 = signal_kwargs_h1.copy()
    signal_kwargs_l1['noiseSFTs'] = ";".join(sft_path_l1)
    signal_kwargs_l1['h0'] = np.min(sqrtSXs_l1) / signal_depth
    writer_h1 = pyfstat.Writer(**signal_kwargs_l1)
    delete_queue.append(
        Path(f'{signal_kwargs_l1["outdir"]}/{signal_kwargs_l1["label"]}.cff'))
    delete_queue.append(
        Path(f'{signal_kwargs_l1["outdir"]}/{signal_kwargs_l1["label"]}_timestamps_L1.csv'))
    writer_h1.make_data()
    delete_queue.append(
        Path(writer_h1.sftfilepath))

    freqs_l1, times_l1, sft_data_l1 = get_sft_as_arrays(writer_h1.sftfilepath)

    for path in delete_queue:
        if path.exists():
            path.unlink()
    
    freqs = freqs_h1
    times = {'H1': times_h1['H1'], 'L1': times_l1['L1']}
    sft_data = {'H1': sft_data_h1['H1'], 'L1': sft_data_l1['L1']}
    slice_start = np.random.randint(90, 270)
    sft_crop = {}
    for d in ['H1', 'L1']:
        sft_crop[d] = sft_data[d][slice_start:slice_start+360]
    freqs = freqs[slice_start:slice_start+360]
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
    num_segment = np.random.randint(1, SEGMENT_MAX, size=2)
    segment_indice_h1 = make_segment(timestamps_h1, num_segment[0])
    segment_indice_l1 = make_segment(timestamps_l1, num_segment[1])

    sft_path_h1 = []
    noise_kwargs_h1 = {
        "outdir": 'pyfstat',
        "detectors": "H1",
        "SFTWindowType": "tukey",
        "SFTWindowBeta": 0.01,
        "Band": 0.4,
        "F0": signal_center,
    }
    delete_queue = []
    for seg_i, (seg_sta, seg_end) in enumerate(
        zip(segment_indice_h1[:-1], segment_indice_h1[1:])):
        noise_kwargs_seg = noise_kwargs_h1.copy()
        noise_kwargs_seg['label'] = f'signal{idx}_h1_{seg_i}'
        noise_kwargs_seg['timestamps'] = timestamps_h1[seg_sta:seg_end]
        noise_kwargs_seg['sqrtSX'] = np.random.uniform(SX_MIN, SX_MAX)
        writer_h1 = pyfstat.Writer(**noise_kwargs_seg)
        writer_h1.make_data()
        sft_path_h1.append(writer_h1.sftfilepath)
        delete_queue.append(
            Path(writer_h1.sftfilepath))
        delete_queue.append(
            Path(f'{noise_kwargs_seg["outdir"]}/{noise_kwargs_seg["label"]}.cff'))
        delete_queue.append(
            Path(f'{noise_kwargs_seg["outdir"]}/{noise_kwargs_seg["label"]}_timestamps_H1.csv'))
   
    freqs_h1, times_h1, sft_data_h1 = get_sft_as_arrays(";".join(sft_path_h1))

    sft_path_l1 = []
    noise_kwargs_l1 = noise_kwargs_h1.copy()
    noise_kwargs_l1['detectors'] = 'L1'
    for seg_i, (seg_sta, seg_end) in enumerate(
        zip(segment_indice_l1[:-1], segment_indice_l1[1:])):
        noise_kwargs_seg = noise_kwargs_l1.copy()
        noise_kwargs_seg['label'] = f'signal{idx}_l1_{seg_i}'
        noise_kwargs_seg['timestamps'] = timestamps_l1[seg_sta:seg_end]
        noise_kwargs_seg['sqrtSX'] = np.random.uniform(SX_MIN, SX_MAX)
        writer_h1 = pyfstat.Writer(**noise_kwargs_seg)
        writer_h1.make_data()
        sft_path_l1.append(writer_h1.sftfilepath)
        delete_queue.append(
            Path(writer_h1.sftfilepath))
        delete_queue.append(
            Path(f'{noise_kwargs_seg["outdir"]}/{noise_kwargs_seg["label"]}.cff'))
        delete_queue.append(
            Path(f'{noise_kwargs_seg["outdir"]}/{noise_kwargs_seg["label"]}_timestamps_L1.csv'))

    freqs_l1, times_l1, sft_data_l1 = get_sft_as_arrays(";".join(sft_path_l1))
    
    for path in delete_queue:
        if path.exists():
            path.unlink()
    
    freqs = freqs_h1
    times = {'H1': times_h1['H1'], 'L1': times_l1['L1']}
    sft_data = {'H1': sft_data_h1['H1'], 'L1': sft_data_l1['L1']}
    slice_start = np.random.randint(90, 270)
    sft_crop = {}
    for d in ['H1', 'L1']:
        sft_crop[d] = sft_data[d][slice_start:slice_start+360]
    freqs = freqs[slice_start:slice_start+360]
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
        try:
            results, metadata = make_signal(index, test_timestamps)
        except Exception as e:
            print(index)
            print(e)
            return {}
        target = 1
    else:
        try:
            results, metadata = make_noise(index, test_timestamps)
        except Exception as e:
            print(index)
            print(e)
            return {}
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
    with open('input/g2net-detecting-continuous-gravitational-waves/test_gaps.pickle', 'rb') as f:
        ts_with_gaps = pickle.load(f)
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
    pd.DataFrame(metadata).dropna(subset='id').reset_index(drop=True).to_csv(
        f'input/g2net-detecting-continuous-gravitational-waves/{DATASET}.csv', index=False)
