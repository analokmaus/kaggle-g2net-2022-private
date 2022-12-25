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


DATASET = 'v18s'
ITERATION = 5
NUM_WORKERS = 40
NUM_BUCKETS = 128
REF_SX = 5e-24
F1_MIN, F1_MAX = -12, -8
DP_MIN, DP_MAX = 25,  50
TEST_DIR = Path('input/g2net-detecting-continuous-gravitational-waves/test/')
TEST_PATH = Path('input/g2net-detecting-continuous-gravitational-waves/sample_submission.csv')
EXPORT_DIR = Path(f'input/g2net-detecting-continuous-gravitational-waves/{DATASET}/')
EXPORT_DIR.mkdir(exist_ok=True)


def to_spectrogram(sfts):
    return sfts.real ** 2 + sfts.imag ** 2


def make_signal(gid):
    TMP_DIR = Path(f'pyfstat_tmp_{DATASET}/{gid}/')
    signal_depth = np.random.uniform(DP_MIN, DP_MAX)
    # load test data
    with open(TEST_DIR/f'{gid}.pickle', 'rb') as f:
        test = pickle.load(f)
        sft_h1, ts_h1 = test[gid]['H1']['SFTs'], test[gid]['H1']['timestamps_GPS']
        sft_l1, ts_l1 = test[gid]['L1']['SFTs'], test[gid]['L1']['timestamps_GPS']
        freqs = test[gid]['frequency_Hz']

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
    writer = pyfstat.Writer(**writer_kwargs, **signal_kwargs)
    writer.make_data()
    _, ts, sft = get_sft_as_arrays(writer.sftfilepath)
    sft['H1'] = sft['H1'][90:450, :]
    shutil.rmtree(TMP_DIR)
    return ts, sft, dict(
        signal_kwargs, **{'signal_depth': signal_depth}
    )


def generate_sample(index, test, iteration=0):
    offset = iteration * len(test)
    np.random.RandomState(offset+index)
    np.random.seed(offset+index)
    test_id = test['id'].values[index]
    try:
        ts, sft, signal_info = make_signal(test_id)
    except Exception as e:
        print(index)
        print(e)
        return {}
    with open(EXPORT_DIR/f'{test_id}_{iteration}.pickle', 'wb') as f:
        pickle.dump({'sft': sft['H1'], 'timestamps': ts['H1']}, f)
    signal_info['id'] = f'{test_id}_{iteration}'
    signal_info['base_id'] = test_id
    return signal_info


if __name__ == '__main__':
    test = pd.read_csv(TEST_PATH)
    # generate samples
    metadata = []
    for it in range(ITERATION):
        with Pool(NUM_WORKERS) as p:
            signal_info = p.map(
                partial(generate_sample, test=test, iteration=it), 
                range(len(test)))
        metadata.extend(signal_info)
    pd.DataFrame(metadata).dropna(subset='id').reset_index(drop=True).to_csv(
        f'input/g2net-detecting-continuous-gravitational-waves/{DATASET}.csv', index=False)
