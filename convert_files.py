# %%
import h5py
from pathlib import Path
import numpy as np
import pickle

# %%
def convert_files_in_folder(target_dir):
    target_files = list(target_dir.glob('*.hdf5'))
    for path in target_files:
        with h5py.File(path, 'r') as f:
            gid = path.stem
            data = {
                gid: {
                    'H1': {
                        'SFTs': np.array(f[gid]['H1']['SFTs']),
                        'timestamps_GPS': np.array(f[gid]['H1']['timestamps_GPS']),
                    },
                    'L1': {
                        'SFTs': np.array(f[gid]['L1']['SFTs']),
                        'timestamps_GPS': np.array(f[gid]['L1']['timestamps_GPS']),
                    },
                    'frequency_Hz': np.array(f[gid]['frequency_Hz'])
                }
            }
            with open(path.parent/f'{path.stem}.pickle', 'wb') as g:
                pickle.dump(data, g)
        path.unlink()

# %%
# convert_files_in_folder(Path('input/g2net-detecting-continuous-gravitational-waves/test'))
convert_files_in_folder(Path('input/g2net-detecting-continuous-gravitational-waves/train'))
# %%
# convert_files_in_folder(Path('input/g2net-detecting-continuous-gravitational-waves/v0/'))


