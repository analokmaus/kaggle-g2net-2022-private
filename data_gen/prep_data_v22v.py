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
import sys
import os
import shutil


DATASET = 'v22v'
NUM_WORKERS = 40
POSITIVE_P = 2/3
SHIFT_RANGE = (-150, 150)
ARTIFACT_NSIGMA = 6
REF_SX = 5e-24
F1_MIN, F1_MAX = -12, -8
DP_MIN, DP_MAX = 25, 50
TEST_DIR = Path('input/g2net-detecting-continuous-gravitational-waves/test/')
TEST_PATH = Path('input/g2net-detecting-continuous-gravitational-waves/sample_submission.csv')
TEST_STATS_PATH = Path('input/signal_stat.pickle')
EXPORT_DIR = Path(f'input/g2net-detecting-continuous-gravitational-waves/{DATASET}/')
EXPORT_DIR.mkdir(exist_ok=True)


def load_test_spec(gid, use_complex=False):
    fname = Path(f'input/g2net-detecting-continuous-gravitational-waves/test/{gid}.pickle')
    with open(fname, 'rb') as fp:
        f = pickle.load(fp)
        sig_h1, time_h1 = f[gid]['H1']['SFTs']*1e22, np.array(f[gid]['H1']['timestamps_GPS'])
        sig_l1, time_l1 = f[gid]['L1']['SFTs']*1e22, np.array(f[gid]['L1']['timestamps_GPS'])
    if use_complex:
        return sig_h1, sig_l1
    else:
        spec_h1 = sig_h1.real ** 2 + sig_h1.imag ** 2
        spec_l1 = sig_l1.real ** 2 + sig_l1.imag ** 2
        return spec_h1, spec_l1


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


def generate_from_stat_complex(gid, signal_stat, artifact_sigma=ARTIFACT_NSIGMA):
    metadata = {
        'id': gid, 
        'nonstationary_H1': 0, 
        'nonstationary_L1': 0,
        'artifact_H1': 0, 
        'artifact_L1': 0,
    }
    asd_time_h1, asd_freq_h1, time_h1 = signal_stat[gid]['H1']
    asd_time_l1, asd_freq_l1, time_l1 = signal_stat[gid]['L1']
    freqs = signal_stat[gid]['freq']
    if asd_time_h1.max() > 3:
        metadata['nonstationary_H1'] = 1
    if asd_time_l1.max() > 3:
        metadata['nonstationary_L1'] = 1
    if asd_freq_h1.max() > 5 or asd_freq_h1.max() > 5:
        metadata['artifact_H1'] = 1
        sft_gen_h1 = reconstruct_from_stat_complex(asd_time_h1.clip(0, 5))
        sft_org = load_test_spec(gid, use_complex=True)[0]
        artifact_map = extract_artifact(sft_org, artifact_sigma)
        sft_gen_h1[np.where(artifact_map)] = sft_org[np.where(artifact_map)]
    else:
        sft_gen_h1 = reconstruct_from_stat_complex(asd_time_h1.clip(0, 5))
       
    if asd_freq_l1.max() > 5 or asd_freq_l1.max() > 5:
        metadata['artifact_L1'] = 1
        sft_gen_l1 = reconstruct_from_stat_complex(asd_time_l1.clip(0, 5))
        sft_org = load_test_spec(gid, use_complex=True)[1]
        artifact_map = extract_artifact(sft_org, artifact_sigma)
        sft_gen_l1[np.where(artifact_map)] = sft_org[np.where(artifact_map)]
    else:
        sft_gen_l1 = reconstruct_from_stat_complex(asd_time_l1.clip(0, 5))
    return (sft_gen_h1, sft_gen_l1, time_h1, time_l1, freqs), metadata


def generate_signal(gid, ts_h1, ts_l1, freqs):
    TMP_DIR = Path(f'pyfstat_tmp_{DATASET}/{gid}/')
    signal_depth = np.random.uniform(DP_MIN, DP_MAX)

    writer_kwargs = {
        "outdir": str(TMP_DIR),
        "label": 'signal',
        "tstart": min(ts_h1[0], ts_l1[0]),
        "duration": 4 * 32 * 86400,
        "detectors": "H1",
        "sqrtSX": 0,
        "Tsft": 1800,
        "SFTWindowType": "tukey",
        "SFTWindowBeta": 0.01,
        "Band": 0.3
    }
    signal_kwargs = {
        "F0": freqs[0],
        "F1": np.random.choice([-1, 1], p=[0.8, 0.2]) * (10 ** np.random.uniform(F1_MIN, F1_MAX)),
        "F2": 0,
        "Alpha": np.random.uniform(0, math.pi * 2),
        "Delta": np.random.uniform(-math.pi/2, math.pi/2),
        "h0": REF_SX / signal_depth,
        "cosi": np.random.uniform(-1, 1),
        "psi": np.random.uniform(-math.pi/4, math.pi/4),
        "phi": np.random.uniform(0, math.pi*2),
    }
    try:
        writer = pyfstat.Writer(**writer_kwargs, **signal_kwargs)
        writer.make_data()
    except:
        print('GENERATION FAILED', signal_kwargs)
        return None, None
    _, ts, sft = get_sft_as_arrays(writer.sftfilepath)
    shutil.rmtree(TMP_DIR)
    sft = sft['H1'][90:450, :]
    return sft, dict(
        signal_kwargs, **{'signal_depth': signal_depth}
    )


def generate_sample(index, test, test_stat):
    np.random.RandomState(index)
    np.random.seed(index)
    test_id = test['id'].values[index]
    (sft_gen_h1, sft_gen_l1, time_h1, time_l1, freqs), metadata = generate_from_stat_complex(test_id, signal_stat=test_stat)
    if np.random.random() < POSITIVE_P:
        sft_signal, metadata_signal = generate_signal(
            test_id, time_h1, time_l1, freqs)
        if sft_signal is not None:
            sft_signal *= 1e22
            shift_y = np.random.randint(*SHIFT_RANGE)
            sft_signal= np.roll(sft_signal, shift_y, axis=0)
            if shift_y > 0:
                sft_signal[:shift_y, :] = 0
            else:
                sft_signal[shift_y:, :] = 0
            ref_time = min(time_h1.min(), time_l1.min())
            frame_h1 = ((time_h1 - ref_time) / 1800).round().astype(np.uint64)
            frame_l1 = ((time_l1 - ref_time) / 1800).round().astype(np.uint64)
            sft_gen_h1 += sft_signal[:, frame_h1]
            sft_gen_l1 += sft_signal[:, frame_l1]
            metadata.update(metadata_signal)
            metadata['target'] = 1
        else:
            metadata['target'] = 0
    else:
        metadata['target'] = 0

    spec_gen_h1 = sft_gen_h1.real**2 + sft_gen_h1.imag**2
    spec_gen_l1 = sft_gen_l1.real**2 + sft_gen_l1.imag**2
    data = {
        'H1': {
            'spectrogram': spec_gen_h1.astype(np.float16),
            'timestamps': time_h1
        },
        'L1': {
            'spectrogram': spec_gen_l1.astype(np.float16),
            'timestamps': time_l1
        },
        'frequency': freqs
    }
    with open(EXPORT_DIR/f'{test_id}_gen.pickle', 'wb') as f:
        pickle.dump(data, f)
    metadata['id'] = f'{test_id}_gen'
    metadata['path'] = str(EXPORT_DIR/f'{test_id}_gen.pickle')
    return metadata


if __name__ == '__main__':
    test = pd.read_csv(TEST_PATH).head(1000)
    with open(TEST_STATS_PATH, 'rb') as f: 
        test_stat = pickle.load(f)
    # generate samples
    with Pool(NUM_WORKERS) as p:
        metadata = p.map(
            partial(generate_sample, test=test, test_stat=test_stat), range(len(test)))
    pd.DataFrame(metadata).dropna(subset='id').reset_index(drop=True).to_csv(
        f'input/g2net-detecting-continuous-gravitational-waves/{DATASET}.csv', index=False)