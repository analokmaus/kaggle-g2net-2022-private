import pickle
from prep_data_v18v import to_spectrogram
from transforms import adaptive_resize
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool


def get_metadata(inputs):
    gid, path = inputs
    with open(path, 'rb') as f:
        data = pickle.load(f)
        data = data[list(data.keys())[0]]
    freq = data['frequency_Hz'].mean()
    return {
        'id': gid, 
        'freq': freq}


valid = pd.read_csv('input/g2net-detecting-continuous-gravitational-waves/concat_v18n1_v18n2.csv')
with Pool(40) as p:
    metadata = p.map(
        get_metadata, 
        list(zip(valid['id'].values, valid['path'].values)))
metadata = pd.DataFrame(metadata)

valid.merge(metadata, on='id', how='left').to_csv(
    'input/g2net-detecting-continuous-gravitational-waves/concat_v18n1_v18n2.csv', index=False)