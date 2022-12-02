import argparse
from pathlib import Path
from pprint import pprint
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.utils.data as D
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import pickle
import traceback
import albumentations as A

from kuma_utils.torch import TorchTrainer, TorchLogger
from kuma_utils.torch.utils import get_time, seed_everything, fit_state_dict

from configs import *
from utils import print_config, notify_me
from training_extras import make_tta_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--silent", action='store_true')
    parser.add_argument("--progress_bar", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--wait", type=int, default=0,
                        help="time (sec) to wait before execution")
    opt = parser.parse_args()
    pprint(opt)

    ''' Configure hardware '''


    ''' Configure path '''
    cfg = eval(opt.config)
    assert cfg.pseudo_labels is None
    export_dir = Path('results') / cfg.name
    export_dir.mkdir(parents=True, exist_ok=True)

    ''' Configure logger '''
    log_items = [
        'epoch', 'train_loss', 'train_metric', 'train_monitor', 
        'valid_loss', 'valid_metric', 'valid_monitor', 
        'learning_rate', 'early_stop'
    ]
    if opt.debug:
        log_items += ['gpu_memory']
    else:
        logger_path = f'{cfg.name}_inference_{get_time("%y%m%d%H%M")}.log'
    LOGGER = TorchLogger(
        export_dir / logger_path, 
        log_items=log_items, file=not opt.silent
    )
    if opt.wait > 0:
        LOGGER(f'Waiting for {opt.wait} sec.')
        time.sleep(opt.wait)

    ''' Prepare data '''
    seed_everything(cfg.seed, cfg.deterministic)
    print_config(cfg, LOGGER)
    test = pd.read_csv(cfg.test_path)

    '''
    Inference
    '''
    test_data = cfg.dataset(
        df=test, data_dir=cfg.test_dir,
        transforms=cfg.transforms['tta'], is_test=True, **cfg.dataset_params)
    test_loader = D.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=opt.num_workers, pin_memory=False)

    LOGGER(f'===== INFERENCE =====')
    model = cfg.model(**cfg.model_params)
    checkpoint = torch.load(export_dir/f'fold0.pt', 'cpu')
    fit_state_dict(checkpoint['model'], model)
    model.load_state_dict(checkpoint['model'])
    del checkpoint; gc.collect()
    if cfg.parallel == 'ddp':
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model.eval()

    predictions = []
    predictions_tta = []
    drop_flag = []
    for idx, (specs, _) in enumerate(test_loader):
        specs = specs.cuda() # (n, ch, f, t)

        # IMPORTANT IGNORE IMAGES WITH ANOMALY
        for i in range(specs.shape[0]): 
            if specs[i].max() > 2.0:
                specs[i, torch.argmax(torch.amax(specs[i], dim=(1,2)))] = 0 # drop single image with anomaly
                drop_flag.append(1)
            else:
                drop_flag.append(0)
        # 

        specs_aug1 = torch.flip(specs, (2,))
        specs_aug2 = torch.flip(specs, (3,))

        with torch.no_grad():
            pred0 = model(specs).cpu().numpy()
            pred1 = model(specs_aug1).cpu().numpy()
            pred2 = model(specs_aug2).cpu().numpy()
        
        predictions.append(pred0)
        predictions_tta.append((pred0 + pred1 + pred2) / 3)
        if idx % 100 == 1:
            LOGGER(f'{idx*cfg.batch_size} sample done.')
        
    torch.cuda.empty_cache()
    predictions = np.concatenate(predictions, axis=0)
    predictions_tta = np.concatenate(predictions_tta, axis=0)
    drop_flag = np.array(drop_flag)

    np.save(export_dir/'predictions_wo_anomaly', predictions)
    np.save(export_dir/'predictions_tta_wo_anomaly', predictions_tta)
    np.save(export_dir/'drop_flag_anomaly', drop_flag)

    LOGGER(f'===== Done =====')
