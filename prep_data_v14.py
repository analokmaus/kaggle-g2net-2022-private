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


DATASET = 'v14'
NUM_WORKERS = 40
NUM_BUCKETS = 128
REF_SX = 5e-24
F1_MIN, F1_MAX = -11, -8 # new sampling
DP_MIN, DP_MID, DP_MAX = 10, 30, 50
C_SQRSX = 26.5
TEST_DIR = Path('input/g2net-detecting-continuous-gravitational-waves/test/')
TEST_PATH = Path('input/test_metadata.csv')
EXPORT_DIR = Path(f'input/g2net-detecting-continuous-gravitational-waves/{DATASET}/')
EXPORT_DIR.mkdir(exist_ok=True)


class NoPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def bucketize_real_noise_asd(sfts, ts, buckets=256):
    bucket_size = (ts.max() - ts.min()) // buckets
    idx = np.searchsorted(ts, [ts[0] + bucket_size * i for i in range(buckets)])
    global_noise_amp = np.mean(np.abs(sfts))
    return np.array([
        np.mean(np.abs(i)) if i.shape[1] > 0 else global_noise_amp for i in np.array_split(sfts, idx[1:], axis=1)]), bucket_size


def make_data(gid, num_buckets=128, target='negative'):
    PROJ_DIR = Path(f'input/g2net-detecting-continuous-gravitational-waves/template_{num_buckets}/{gid}')
    PROJ_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR = Path(f'pyfstat_tmp_{DATASET}/{gid}/')
    # load test data
    with open(TEST_DIR/f'{gid}.pickle', 'rb') as f:
        test = pickle.load(f)
        sft_h1, ts_h1 = test[gid]['H1']['SFTs'], test[gid]['H1']['timestamps_GPS']
        sft_l1, ts_l1 = test[gid]['L1']['SFTs'], test[gid]['L1']['timestamps_GPS']
        freqs = test[gid]['frequency_Hz']

    asd_h1, bs_h1 = bucketize_real_noise_asd(sft_h1, ts_h1, num_buckets)
    asd_l1, bs_l1 = bucketize_real_noise_asd(sft_l1, ts_l1, num_buckets)

    with NoPrint():
        noise_kwargs_h1 = {
            "outdir": str(TMP_DIR),
            "Tsft": 1800,
            "F0": np.mean(freqs),
            "detectors": "H1",
            "SFTWindowType": "tukey",
            "SFTWindowBeta": 0.01,
            "Band": 0.4,
            "duration": bs_h1,
        }
        if (PROJ_DIR/'sft_list_h1.csv').exists():
            sft_paths_h1 = pd.read_csv(PROJ_DIR/'sft_list_h1.csv')['path'].tolist()
        else:
            tmp_paths = []
            sft_paths_h1 = []
            for segment in range(num_buckets):
                args = noise_kwargs_h1.copy()
                args["label"] = f"h1_segment_{segment}"
                args["sqrtSX"] = asd_h1[segment] / C_SQRSX
                args["tstart"] = ts_h1[0] + segment * bs_h1
                writer = pyfstat.Writer(**args)
                writer.make_data()
                tmp_paths.append(writer.sftfilepath)
            for path in tmp_paths:
                shutil.move(Path(path), PROJ_DIR)
                sft_paths_h1.append(str(PROJ_DIR/Path(path).name))
            pd.DataFrame({'path': sft_paths_h1}).to_csv(PROJ_DIR/'sft_list_h1.csv', index=False)

        if (PROJ_DIR/'sft_list_l1.csv').exists():
            sft_paths_l1 = pd.read_csv(PROJ_DIR/'sft_list_l1.csv')['path'].tolist()
        else:
            noise_kwargs_l1 = noise_kwargs_h1.copy()
            noise_kwargs_l1["detectors"] = "L1"
            noise_kwargs_l1["duration"] = bs_l1
            tmp_paths = []
            sft_paths_l1 = []
            for segment in range(num_buckets):
                args = noise_kwargs_l1.copy()
                args["label"] = f"l1_segment_{segment}"
                args["sqrtSX"] = asd_l1[segment] / C_SQRSX
                args["tstart"] = ts_l1[0] + segment * bs_l1
                writer = pyfstat.Writer(**args)
                writer.make_data()
                tmp_paths.append(writer.sftfilepath)
            for path in tmp_paths:
                shutil.move(Path(path), PROJ_DIR)
                sft_paths_l1.append(str(PROJ_DIR/Path(path).name))
            shutil.rmtree(TMP_DIR)
            pd.DataFrame({'path': sft_paths_l1}).to_csv(PROJ_DIR/'sft_list_l1.csv', index=False)
    
    if target == 'negative':
        signal_depth = 1000
        freqs_h1, times_h1, sft_data_h1 = get_sft_as_arrays(";".join(sorted(sft_paths_h1)))
        freqs_l1, times_l1, sft_data_l1 = get_sft_as_arrays(";".join(sorted(sft_paths_l1)))
    else:
        with NoPrint():
            signal_kwargs = {
                "outdir": str(TMP_DIR),
                "label": f'h1_signal',
                "F0": np.mean(freqs),
                "F1": np.random.choice([-1, 1]) * (10 ** np.random.uniform(F1_MIN, F1_MAX)),
                "F2": 0,
                "Alpha": np.random.uniform(0, math.pi * 2),
                "Delta": np.random.uniform(-math.pi/2, math.pi/2),
                # "h0": noise_kwargs_h1['sqrtSX'] / signal_depth,
                "cosi": np.random.uniform(-1, 1),
                "psi": np.random.uniform(-math.pi/4, math.pi/4),
                "phi": np.random.uniform(0, math.pi*2),
                "SFTWindowType": "tukey",
            }
            if target == 'strong': # 
                signal_depth = np.random.uniform(DP_MIN, DP_MID)
            elif target == 'weak': 
                signal_depth = np.random.uniform(DP_MID, DP_MAX)

            # H1
            signal_variety = np.random.uniform(0.9, 1.1)
            signal_kwargs['label'] = f'l1_signal'
            signal_kwargs['h0'] = REF_SX * signal_variety / signal_depth
            signal_kwargs['noiseSFTs'] = ";".join(sorted(sft_paths_h1))
            writer = pyfstat.Writer(**signal_kwargs)
            writer.make_data()
            freqs_h1, times_h1, sft_data_h1 = get_sft_as_arrays(writer.sftfilepath)
            # L1
            signal_variety = np.random.uniform(0.9, 1.1)
            signal_kwargs['h0'] = REF_SX * signal_variety / signal_depth
            signal_kwargs['noiseSFTs'] = ";".join(sorted(sft_paths_l1))
            writer = pyfstat.Writer(**signal_kwargs)
            writer.make_data()
            freqs_l1, times_l1, sft_data_l1 = get_sft_as_arrays(writer.sftfilepath)
            shutil.rmtree(TMP_DIR)

    use_idx_h1 = np.sort(np.random.choice(np.arange(len(times_h1['H1'])), len(ts_h1), replace=False))
    use_idx_l1 = np.sort(np.random.choice(np.arange(len(times_l1['L1'])), len(ts_l1), replace=False))
    times = {'H1': times_h1['H1'][use_idx_h1], 'L1': times_l1['L1'][use_idx_l1]}
    sft_data = {'H1': sft_data_h1['H1'][:, use_idx_h1], 'L1': sft_data_l1['L1'][:, use_idx_l1]}
    freqs = freqs_h1
    slice_start = np.random.randint(90, 270)
    sft_crop = {}
    for d in ['H1', 'L1']:
        sft_crop[d] = sft_data[d][slice_start:slice_start+360]
    freqs = freqs[slice_start:slice_start+360]
    fname = f'{gid}_{target}'
    target_val = 0 if target == 'negative' else 1
    return {fname: {
                'H1': {
                    'SFTs': sft_crop['H1'], 
                    'timestamps_GPS': times['H1']
                },
                'L1': {
                    'SFTs': sft_crop['L1'], 
                    'timestamps_GPS': times['L1']
                },
            'frequency_Hz': freqs}}, {
            'id': fname,
            'base_id': gid, 
            'signal_depth': signal_depth,
            'target': target_val
        }


def generate_sample(index, test, target):
    np.random.RandomState(index)
    np.random.seed(index)
    test_id = test['id'].values[index]
    try:
        results, metadata = make_data(test_id, NUM_BUCKETS, target)
    except Exception as e:
        print(index)
        print(e)
        return {}
    instance_id = metadata['id']
    with open(EXPORT_DIR/f'{instance_id}.pickle', 'wb') as f:
        pickle.dump(results, f)
    return metadata


if __name__ == '__main__':
    test = pd.read_csv(TEST_PATH)
    test = test.query('max_h1 <= 150 and max_l1 <= 150') # remove noise
    # generate samples
    with Pool(NUM_WORKERS) as p:
        metadata_neg = p.map(
            partial(generate_sample, test=test, target='negative'), 
            range(len(test)))
    with Pool(NUM_WORKERS) as p:
        metadata_weak = p.map(
            partial(generate_sample, test=test, target='weak'), 
            range(len(test)))
    with Pool(NUM_WORKERS) as p:
        metadata_strong = p.map(
            partial(generate_sample, test=test, target='strong'), 
            range(len(test)))
    metadata = metadata_neg + metadata_weak + metadata_strong
    pd.DataFrame(metadata).dropna(subset='id').reset_index(drop=True).to_csv(
        f'input/g2net-detecting-continuous-gravitational-waves/{DATASET}.csv', index=False)
