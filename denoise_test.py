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

# # shortcut :)
# with open('input/g2net-detecting-continuous-gravitational-waves/leak_0.pickle', 'rb') as handle:
#     clean_data = pickle.load(handle)


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


def get_correlation(x1,x2):
    correlation = (torch.einsum('ki,kj->ij', x1, x2)/x1.shape[0] - \
                   torch.einsum('i,j->ij', x1.mean(0), x2.mean(0)))/ \
                   torch.einsum('i,j->ij', x1.std(0), x2.std(0))
    return correlation


clean_data = {}
TH = 0.95
denoised_target = {}
for index, row in tqdm(df.iterrows(), total=len(df)):
# for index, row in tqdm(df.loc[df['id'].isin(clean_data.keys())].iterrows()):
    # data_src = extract_data_from_hdf5(os.path.join(PATH, idx+'.hdf5')) 
    idx, freq_origin = row[['id','freq']]
    data_src = extract_data_from_hdf5(os.path.join(PATH, idx+'.pickle'))
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
            # data_tgt = extract_data_from_hdf5(os.path.join(PATH, key+'.hdf5'))
            data_tgt = extract_data_from_hdf5(os.path.join(PATH, key+'.pickle'))
            
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

                sample1 = src_H[f1s:f2s, values > TH].mean(1)
                sample2 = tgt_H[f1t:f2t, indices[values > TH]].mean(1)
                diff_raw = sample1 - sample2
                diff_raw_roll10 = sliding_window_view(diff_raw, 10, axis=0).mean(axis=-1)
                negative_peak = np.abs(diff_raw_roll10[diff_raw_roll10 < -1e-4]).sum() > 1.05e-3
                positive_peak = np.abs(diff_raw_roll10[diff_raw_roll10 > 1e-4]).sum() > 1.05e-3
                if negative_peak:
                    if not key in denoised_target.keys():
                        denoised_target[key] = [1]
                    else:
                        denoised_target[key].append(1)
                else:
                    if not key in denoised_target.keys():
                        denoised_target[key] = [0]
                    else:
                        denoised_target[key].append(0)
                if positive_peak:
                    if not idx in denoised_target.keys():
                        denoised_target[idx] = [1]
                    else:
                        denoised_target[idx].append(1)
                else:
                    if not idx in denoised_target.keys():
                        denoised_target[idx] = [0]
                    else:
                        denoised_target[idx].append(0)
                
                # plt.figure(figsize=(12, 5))
                # plt.subplot(1, 2, 1)
                # v = dif_abs[:, :512]
                # v = (v - v.min())/(v.max() - v.min())
                # plt.imshow(v)
                # plt.subplot(1, 2, 2)
                # plt.plot(diff_raw)
                # plt.plot(diff_raw_roll10)
                # # plt.plot(diff_raw_cumsum)
                # plt.title(f'{idx} - {key} {negative_peak} {positive_peak}')
                # # plt.show()
                # plt.savefig(f'input/plot/leak_pos/{idx}-{key}.png', facecolor='white')
                # plt.close()

            if correlation_L.max() > TH:
                values,indices = correlation_L.max(-1)
                dif_abs = (src_L[f1s:f2s, values > TH] - tgt_L[f1t:f2t,indices[values > TH]]).abs()
                min_val = torch.min(buf['L1'][f1s:f2s,values > TH], dif_abs)
                buf['L1'][f1s:f2s,values > TH] = torch.where(buf['L1_mask'][f1s:f2s,values > TH] == 0,
                        dif_abs, min_val)
                buf['L1_mask'][f1s:f2s,values > TH] += 1
                buf['empty'] = False
                buf['denoise_pair_L'].append(key)

            #print(correlation_H.max(),correlation_L.max())
        
    if not buf['empty']:
        del buf['empty']
        buf['coverage_H1'] = (buf['H1_mask'].amax(1) > 0).sum() / 360
        buf['coverage_L1'] = (buf['L1_mask'].amax(1) > 0).sum() / 360
        # buf['signal'] = buf['H1'].max() > 0.01 or buf['L1'].max() > 0.01
        # buf['signal_stats'] = (buf['H1'].sum(), buf['L1'].max())
        clean_data[idx] = buf
        #if buf['H1'].max() > 0.01: break
        #if buf['H1_mask'].max() > 1: break

        # plt.figure(figsize=(16, 12))
        # plt.subplot(2, 2, 1)
        # v = buf['H1_mask']
        # v = (v - v.min())/(v.max() - v.min())
        # plt.imshow(v[:,:512])
        # plt.subplot(2, 2, 2)
        # v = buf['L1_mask']
        # v = (v - v.min())/(v.max() - v.min())
        # plt.imshow(v[:,:512])
        # plt.subplot(2, 2, 3)
        # v = buf['H1']
        # v = (v - v.min())/(v.max() - v.min())
        # plt.imshow(v[:,:512])
        # plt.subplot(2, 2, 4)
        # v = buf['L1']
        # v = (v - v.min())/(v.max() - v.min())
        # plt.imshow(v[:,:512])
        # plt.suptitle(f'{buf["signal"]} / {buf["signal_stat_H"]} / {buf["signal_stat_L"]}')
        # plt.savefig(f'input/plot/leak2/{idx}.png', facecolor='white')

        # freq_mean = buf['H1'][:, :512].mean(1)
        # freq_mask = buf['H1_mask'].amax(1) > 0
        # v = buf['H1'][:, :512]
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(v)
        # plt.subplot(1, 2, 2)
        # plt.plot(freq_mean[freq_mask])
        # plt.axhline(y=freq_mean[freq_mask].mean(), color='r')
        # plt.savefig(f'input/plot/leak_pos/{idx}.png', facecolor='white')

        # plt.close()

denoised_target = {k: max(v) for k, v in denoised_target.items()}
for k, v in denoised_target.items():
    if k in clean_data.keys():
        clean_data[k]['label'] = v

denoised_target = pd.DataFrame.from_dict(denoised_target, orient='index').reset_index()
denoised_target.columns = ['id', 'target']
denoised_target.to_csv('input/denoised_target_2.csv', index=False)

with open(OUT, 'wb') as handle:
    pickle.dump(clean_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


from configs import *
from configs_chris import *
from kuma_utils.utils import sigmoid


with open(OUT, 'rb') as handle:
    clean_data = pickle.load(handle)


leak_target = pd.read_csv('input/denoised_target_2.csv')
leak_target['coverage'] = leak_target['id'].apply(lambda x: 0.5*(clean_data[x]['coverage_H1']+clean_data[x]['coverage_L1']).item())


def mask_image(x_tensor, mask_tensor): # (B, C, H, W)
    x_tensor[:, :, mask_tensor, :] = 0
    return x_tensor


def infer(x_aug0):
    with torch.no_grad():
        x_aug1 = torch.flip(x_aug0, (2,))
        x_aug2 = torch.flip(x_aug0, (3,))
        y_aug0 = model(x_aug0)
        y_aug1 = model(x_aug1)
        y_aug2 = model(x_aug2)
        y = (y_aug0 + y_aug1 + y_aug2) / 3
    return y.item()


cfg = Ds20l()
model = cfg.model(**cfg.model_params)
checkpoint = torch.load(f'results/{cfg.name}/fold0.pt', 'cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
test = pd.read_csv('input/g2net-detecting-continuous-gravitational-waves/test.csv')
test = test.merge(leak_target[['id']], on='id', how='right')


dataset = cfg.dataset(
    df=test,
    data_dir=cfg.train_dir,
    transforms=cfg.transforms['test'],
    is_test=True,
    **dict(cfg.dataset_params)
)


updated_target = []
for i, (gid, target) in enumerate(tqdm(leak_target[['id', 'target']].values)):
    if target == 1:
        updated_target.append({'id': gid, 'target': 1})
        continue
    x_mask = mask_image(
        dataset[i][0][None, :, :, :], 
        clean_data[gid]['H1_mask'].sum(1) > 0)
    y_mask = sigmoid(infer(x_mask))
    updated_target.append({'id': gid, 'target': y_mask})
    print(gid, y_mask)
updated_target = pd.DataFrame(updated_target)


updated_target.to_csv('input/denoised_target_negative_2.csv', index=False)