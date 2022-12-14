import os
import numpy as np
from tqdm.notebook import tqdm
import random
import pickle
import cv2
import shutil
from scipy.stats import binned_statistic
from functools import partial
from scipy import stats
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings('ignore')

import pyfstat
from pyfstat.utils import get_sft_as_arrays

import logging
logging.getLogger("pyfstat").setLevel(logging.CRITICAL)



def gen_data(i, params):
    tmp_path = params['tmp_path'] + f'_{i}'
    writer_kwargs = {
        "label": "single_detector_gaussian_noise",
        "outdir": tmp_path,
        "detectors": "L1,H1",
        "sqrtSX": 0,
        "Tsft": 1800,
        "SFTWindowType": "tukey",
        "SFTWindowBeta": 0.01,
    }

    signal_parameters_generator = pyfstat.AllSkyInjectionParametersGenerator(
        priors={
            "Band": 0.4,
            "F0": {"uniform": {"low": 50.0, "high": 500.1}},
            "F1": lambda: 10**stats.uniform(-12, 4).rvs(),
            "F2": 0,
            "h0": 1e-2,
            **pyfstat.injection_parameters.isotropic_amplitude_priors,
        }, seed=i,
    )
    
    s = random.choice(params['timestamps'])
    sz = params['L']
    writer_kwargs['timestamps'] = {"H1": s['H1_ts'],"L1": s['L1_ts']}
    signal_parameters = signal_parameters_generator.draw()
    writer = pyfstat.Writer(**writer_kwargs, **signal_parameters)
    writer.make_data()
    frequency, timestamps, fourier_data = get_sft_as_arrays(writer.sftfilepath)
    frequency,fourier_data = frequency[1:], {'H1':fourier_data['H1'][1:], 'L1':fourier_data['L1'][1:]}
    
    n_statH = binned_statistic(timestamps['H1'], np.ones((1,len(timestamps['H1']))), statistic='sum', bins=sz, 
        range=(max(timestamps['H1'].min(),timestamps['L1'].min()),
        min(timestamps['H1'].max(),timestamps['L1'].max())))
    n_statL = binned_statistic(timestamps['L1'], np.ones((1,len(timestamps['L1']))), statistic='sum', bins=sz, 
        range=(max(timestamps['H1'].min(),timestamps['L1'].min()),
        min(timestamps['H1'].max(),timestamps['L1'].max())))
    n_statH = np.nan_to_num(n_statH.statistic)[0].astype(np.long)
    n_statL = np.nan_to_num(n_statL.statistic)[0].astype(np.long)

    mean_statH = binned_statistic(timestamps['H1'], (np.abs(fourier_data['H1'])**2), statistic='mean', bins=sz, 
        range=(max(timestamps['H1'].min(),timestamps['L1'].min()),
        min(timestamps['H1'].max(),timestamps['L1'].max())))
    mean_statL = binned_statistic(timestamps['L1'], (np.abs(fourier_data['L1'])**2), statistic='mean', bins=sz, 
        range=(max(timestamps['H1'].min(),timestamps['L1'].min()),
        min(timestamps['H1'].max(),timestamps['L1'].max())))
    mean_statH = np.nan_to_num(np.transpose(mean_statH.statistic,(0,1))).astype(np.float16())[90:-90]
    mean_statL = np.nan_to_num(np.transpose(mean_statL.statistic,(0,1))).astype(np.float16())[90:-90]
    
    shutil.rmtree(tmp_path)
    return {'H1':mean_statH,'L1':mean_statL,'H1_ts':n_statH, 'L1_ts':n_statL}
    

#Mount RAM disk
#sudo mkdir /mnt/ramdisk
#sudo mount -t tmpfs -o size=2048m tmpfs /mnt/ramdisk

if __name__ == '__main__':
    OUT = 'gwaves_clean_v5.pickle'
    TIMESTAMPS = 'timestamps_all.pickle'
    sz = 128
    N = 32768

    with open(TIMESTAMPS, 'rb') as handle:
        timestamps_all = pickle.load(handle)
    timestamps_all = [timestamps for timestamps in timestamps_all \
                  if len(timestamps['L1_ts']) >= 4000 and len(timestamps['H1_ts']) >= 4000]
    params = {'tmp_path':'/mnt/ramdisk/PyFstat_example_data', 'timestamps':timestamps_all, 'L': sz}

    from tqdm.contrib.concurrent import process_map
    results = process_map(partial(gen_data,params=params), range(N), max_workers=cpu_count())
    
    with open(OUT, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

