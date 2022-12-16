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
from timm.models import convert_sync_batchnorm
import multiprocessing
multiprocessing.current_process().authkey = '0'.encode('utf-8')

from kuma_utils.torch import TorchTrainer, TorchLogger
from kuma_utils.torch.utils import get_time, seed_everything, fit_state_dict

from configs import *
from configs_chris import *
from team.chris_model import forward_test_chris
from utils import print_config, notify_me
from training_extras import make_tta_dataloader


def inference(data_loader, tta=False, drop_anomaly=False):
    predictions = []
    predictions_tta = []
    drop_flag = []
    for idx, inputs in enumerate(data_loader):
        if len(inputs) == 2:
            specs, _ = inputs
            specs = specs.cuda() # (n, ch, f, t)
        elif len(inputs) > 2:
            inputs = [input_t.cuda() for input_t in inputs]
            specs = inputs[0]

        if drop_anomaly:
            for i in range(specs.shape[0]):  
                specs_std = specs[i].std(dim=(1, 2))
                specs_min = specs[i].amin(dim=(1, 2))
                specs_max = specs[i].amax(dim=(1, 2))
                peak_sigma = (specs_max - specs_min) / specs_std
                if peak_sigma.amax() > 25.0:
                    specs[i, torch.argmax(peak_sigma)] = 0 # drop single image with anomaly
                    drop_flag.append(1)
                else:
                    drop_flag.append(0)
        else:
            for i in range(specs.shape[0]): 
                drop_flag.append(0)

        if tta:
            specs_aug1 = torch.flip(specs, (2,))
            specs_aug2 = torch.flip(specs, (3,))

        with torch.no_grad():
            if len(inputs) == 2:
                pred0 = model(specs).cpu().numpy()
                if tta:
                    pred1 = model(specs_aug1).cpu().numpy()
                    pred2 = model(specs_aug2).cpu().numpy()
            elif len(inputs) > 2:
                pred0 = forward_test_chris(model, specs, inputs).cpu().numpy()
                if tta:
                    pred1 = forward_test_chris(model, specs_aug1, inputs).cpu().numpy()
                    pred2 = forward_test_chris(model, specs_aug2, inputs).cpu().numpy()
            
        predictions.append(pred0)
        if tta:
            predictions_tta.append((pred0 + pred1 + pred2) / 3)
        
    torch.cuda.empty_cache()
    predictions = np.concatenate(predictions, axis=0)
    if tta:
        predictions_tta = np.concatenate(predictions_tta, axis=0)
    else:
        predictions_tta = predictions
    drop_flag = np.array(drop_flag)
    return predictions, predictions_tta, drop_flag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_limit", type=int, default=0) # in GB
    parser.add_argument("--limit_fold", type=int, default=-1,
                        help="train only specified fold")
    parser.add_argument("--inference", action='store_true',
                        help="inference")
    parser.add_argument("--tta", action='store_true', 
                        help="test time augmentation ")
    parser.add_argument("--drop_anomaly", action='store_true', 
                        help="drop anomaly")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--silent", action='store_true')
    parser.add_argument("--progress_bar", action='store_true')
    parser.add_argument("--skip_existing", action='store_true')
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
    if opt.limit_fold >= 0:
        logger_path = f'{cfg.name}_fold{opt.limit_fold}_{get_time("%y%m%d%H%M")}.log'
    else:
        logger_path = f'{cfg.name}_{get_time("%y%m%d%H%M")}.log'
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
    train = pd.read_csv(cfg.train_path)
    test = pd.read_csv(cfg.test_path)
    if cfg.debug:
        train = train.sample(10000)
        test = test.iloc[:1000]
    train = train.loc[train['target'] != -1]
    splitter = cfg.splitter
    if isinstance(splitter, (str, Path)):
        with open(splitter, 'rb') as f:
            fold_iter = pickle.load(f)
    else:
        if splitter.__class__.__name__ == 'StratifiedKFold':
            fold_iter = list(splitter.split(X=train, y=train['target']))
        elif splitter.__class__.__name__ == 'StratifiedGroupKFold':
            if 'group' not in train.columns:
                train['group'] = train['id'].apply(lambda x: x.split('_')[0])
            fold_iter = list(splitter.split(X=train, y=train['target'], groups=train['group']))
        elif splitter.__class__.__name__ == 'MultilabelStratifiedKFold':
            targets = pd.get_dummies(pd.cut(train['signal_depth'], cfg.depth_bins))
            fold_iter = list(splitter.split(X=train, y=targets))
        with open(export_dir/'folds.pickle', 'wb') as f:
            pickle.dump(fold_iter, f)
    
    '''
    Training
    '''
    scores = []
    manager = multiprocessing.Manager()
    cache = manager.dict()
    cache['size'] = 0
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):
        
        if opt.limit_fold >= 0 and fold != opt.limit_fold:
            continue  # skip fold

        if opt.inference:
            continue

        if opt.skip_existing and (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'checkpoint fold{fold}.pt already exists.')
            continue

        LOGGER(f'===== TRAINING FOLD {fold} =====')

        train_fold = train.iloc[train_idx]
        valid_fold = train.iloc[valid_idx]

        LOGGER(f'train positive: {train_fold.target.values.mean(0)} ({len(train_fold)})')
        LOGGER(f'valid positive: {valid_fold.target.values.mean(0)} ({len(valid_fold)})')

        train_data = cfg.dataset(
            df=train_fold, data_dir=cfg.train_dir,
            transforms=cfg.transforms['train'], is_test=False, 
            **dict(cfg.dataset_params, **{'cache_limit': opt.cache_limit}))
        train_data.cache = cache
        valid_data = cfg.dataset(
            df=valid_fold, data_dir = cfg.train_dir,
            transforms=cfg.transforms['test'], is_test=True, 
            **dict(cfg.dataset_params, **{'cache_limit': 0}))
        valid_data.cache = cache

        train_loader = D.DataLoader(
            train_data, batch_size=cfg.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=False)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=False)

        model = cfg.model(**cfg.model_params)

        # Load snapshot
        if cfg.weight_path is not None:
            if cfg.weight_path.is_dir():
                weight_path = cfg.weight_path / f'fold{fold}.pt'
            else:
                weight_path = cfg.weight_path
            LOGGER(f'{weight_path} loaded.')
            weight = torch.load(weight_path, 'cpu')
            if 'model' in weight.keys():
                weight = weight['model']
            fit_state_dict(weight, model)
            model.load_state_dict(weight, strict=False)
            del weight; gc.collect()

        optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_params)
        scheduler = cfg.scheduler(optimizer, **cfg.scheduler_params)
        FIT_PARAMS = {
            'loader': train_loader,
            'loader_valid': valid_loader,
            'criterion': cfg.criterion,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'scheduler_target': cfg.scheduler_target,
            'batch_scheduler': cfg.batch_scheduler, 
            'num_epochs': cfg.num_epochs,
            'callbacks': deepcopy(cfg.callbacks),
            'hook': cfg.hook,
            'export_dir': export_dir,
            'eval_metric': cfg.eval_metric,
            'monitor_metrics': cfg.monitor_metrics,
            'fp16': cfg.amp,
            'parallel': cfg.parallel,
            'deterministic': cfg.deterministic, 
            'clip_grad': cfg.clip_grad, 
            'max_grad_norm': cfg.max_grad_norm,
            'random_state': cfg.seed,
            'logger': LOGGER,
            'progress_bar': opt.progress_bar, 
            'resume': opt.resume
        }
        if not cfg.debug:
            notify_me(f'[{cfg.name}:fold{opt.limit_fold}]\nTraining started.')
        try:
            trainer = TorchTrainer(model, serial=f'fold{fold}', device=None)
            trainer.ddp_sync_batch_norm = convert_sync_batchnorm
            trainer.ddp_params = dict(broadcast_buffers=True)
            trainer.fit(**FIT_PARAMS)
        except Exception as e:
            err = traceback.format_exc()
            LOGGER(err)
            if not opt.silent:
                notify_me('\n'.join([
                    f'[{cfg.name}:fold{opt.limit_fold}]', 
                    'Training stopped due to:', 
                    f'{traceback.format_exception_only(type(e), e)}'
                ]))
        del model, trainer, train_data, valid_data; gc.collect()
        torch.cuda.empty_cache()


    '''
    Inference
    '''
    predictions = np.full((cfg.cv, len(test), 1), 0.5, dtype=np.float32)
    predictions_tta = np.full((cfg.cv, len(test), 1), 0.5, dtype=np.float32)
    anomaly_flag = np.full((cfg.cv, len(test)), 0, dtype=np.uint8)
    outoffolds = np.full((len(train), 1), 0.5, dtype=np.float32)
    test_data = cfg.dataset(
        df=test, data_dir=cfg.test_dir,
        transforms=cfg.transforms['tta'], is_test=True, 
        **dict(cfg.dataset_params, **{'cache_limit': 0}))
    test_loader = D.DataLoader(
            test_data, batch_size=cfg.batch_size, shuffle=False, 
            num_workers=opt.num_workers, pin_memory=False)
    
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):

        if opt.limit_fold >= 0 and fold != opt.limit_fold:
            continue  # skip fold

        if not (export_dir/f'fold{fold}.pt').exists():
            LOGGER(f'fold{fold}.pt missing. No target to predict.')
            continue

        LOGGER(f'===== INFERENCE FOLD {fold} =====')

        valid_fold = train.iloc[valid_idx]
        valid_data = cfg.dataset(
            df=valid_fold, data_dir = cfg.train_dir,
            transforms=cfg.transforms['tta'], is_test=True, 
            **dict(cfg.dataset_params, **{'cache_limit': opt.cache_limit}))
        valid_data.cache = cache
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=False)

        model = cfg.model(**cfg.model_params)
        checkpoint = torch.load(export_dir/f'fold{fold}.pt', 'cpu')
        fit_state_dict(checkpoint['model'], model)
        model.load_state_dict(checkpoint['model'])
        del checkpoint; gc.collect()
        if cfg.parallel == 'ddp':
            model = convert_sync_batchnorm(model)
        model.cuda()
        model.eval()

        pred, pred_tta, drop_flag = inference(test_loader, opt.tta, opt.drop_anomaly)
        oof, _, _ = inference(valid_loader, False, False)
        predictions[fold] = pred
        predictions_tta[fold] = pred_tta
        anomaly_flag[fold] = drop_flag
        outoffolds[valid_idx] = oof

        del model, valid_data; gc.collect()
        torch.cuda.empty_cache()

    np.save(export_dir/'outoffolds', outoffolds)
    np.save(export_dir/'predictions', predictions)
    if opt.tta:
        np.save(export_dir/'predictions_tta', predictions_tta)
    if opt.drop_anomaly:
        np.save(export_dir/'anomaly_flag', anomaly_flag)
        
    LOGGER(f'scores: {scores}')
    LOGGER(f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}')
    if not cfg.debug:
        notify_me('\n'.join([
            f'[{cfg.name}:fold{opt.limit_fold}]',
            'Training has finished successfully.',
            f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}'
        ]))