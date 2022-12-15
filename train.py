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
from sklearn.metrics import roc_auc_score

from kuma_utils.torch import TorchTrainer, TorchLogger
from kuma_utils.torch.utils import get_time, seed_everything, fit_state_dict

from configs import *
from configs_chris import *
from team.chris_model import forward_test_chris
from utils import print_config, notify_me
from training_extras import make_tta_dataloader


def inference(data_loader, tta=False):
    predictions = []
    predictions_tta = []
    for idx, inputs in enumerate(data_loader):
        if len(inputs) == 2:
            specs, _ = inputs
            specs = specs.cuda() # (n, ch, f, t)
        elif len(inputs) > 2:
            inputs = [input_t.cuda() for input_t in inputs]
            specs = inputs[0]

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
    return predictions, predictions_tta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_limit", type=int, default=0) # in GB
    parser.add_argument("--inference", action='store_true',
                        help="inference")
    parser.add_argument("--tta", action='store_true', 
                        help="test time augmentation ")
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
    valid = pd.read_csv(cfg.valid_path)
    test = pd.read_csv(cfg.test_path)
    if cfg.debug:
        train = train.sample(10000)
        valid = valid.sample(1000)
        test = test.iloc[:1000]
    
    '''
    Training
    '''
    manager = multiprocessing.Manager()
    cache_train = manager.dict()
    cache_train['size'] = 0
    cache_valid = manager.dict()
    cache_valid['size'] = 0
    fold = 0
    
    if opt.inference:
        pass
    elif opt.skip_existing and (export_dir/f'fold{fold}.pt').exists():
        LOGGER(f'checkpoint fold{fold}.pt already exists.')
        pass
    else:
        
        LOGGER(f'===== TRAINING FOLD {fold} =====')

        train_fold = train
        valid_fold = valid

        if 'target' in train_fold.columns:
            LOGGER(f'train positive: {train_fold.target.values.mean(0)} ({len(train_fold)})')
        LOGGER(f'valid positive: {valid_fold.target.values.mean(0)} ({len(valid_fold)})')

        train_data = cfg.dataset(
            df=train_fold, data_dir=cfg.train_dir,
            transforms=cfg.transforms['train'], is_test=False, 
            **dict(cfg.dataset_params, **{'cache_limit': opt.cache_limit}))
        train_data.cache = cache_train
        valid_data = cfg.dataset(
            df=valid_fold, data_dir = cfg.valid_dir,
            transforms=cfg.transforms['test'], is_test=True, 
            **dict(cfg.dataset_params, **{'cache_limit': opt.cache_limit/5}))
        valid_data.cache = cache_valid

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
        try:
            trainer = TorchTrainer(model, serial=f'fold{fold}', device=None)
            trainer.ddp_sync_batch_norm = convert_sync_batchnorm
            trainer.ddp_params = dict(broadcast_buffers=True)
            trainer.fit(**FIT_PARAMS)
        except Exception as e:
            err = traceback.format_exc()
            LOGGER(err)
        del model, trainer, train_data, valid_data; gc.collect()
        torch.cuda.empty_cache()


    '''
    Inference
    '''
    predictions = np.full((len(test), 1), 0.5, dtype=np.float32)
    predictions_tta = np.full((len(test), 1), 0.5, dtype=np.float32)
    outoffolds = np.full((len(valid), 1), 0.5, dtype=np.float32)
    outoffolds_tta = np.full((len(valid), 1), 0.5, dtype=np.float32)
    test_data = cfg.dataset(
        df=test, data_dir=cfg.test_dir,
        transforms=cfg.transforms['tta'], is_test=True, 
        **dict(cfg.dataset_params, **{'cache_limit': 0}))
    test_loader = D.DataLoader(
            test_data, batch_size=cfg.batch_size, shuffle=False, 
            num_workers=opt.num_workers, pin_memory=False)
    valid_data = cfg.dataset(
        df=valid, data_dir = cfg.valid_dir,
        transforms=cfg.transforms['tta'], is_test=True, 
        **dict(cfg.dataset_params, **{'cache_limit': opt.cache_limit/5}))
    valid_data.cache = cache_valid
    valid_loader = D.DataLoader(
        valid_data, batch_size=cfg.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=False)

    if not (export_dir/f'fold{fold}.pt').exists():
        LOGGER(f'fold{fold}.pt missing. No target to predict.')
        pass
    else:

        LOGGER(f'===== INFERENCE FOLD {fold} =====')
        
        model = cfg.model(**cfg.model_params)
        checkpoint = torch.load(export_dir/f'fold{fold}.pt', 'cpu')
        fit_state_dict(checkpoint['model'], model)
        model.load_state_dict(checkpoint['model'])
        del checkpoint; gc.collect()
        if cfg.parallel == 'ddp':
            model = convert_sync_batchnorm(model)
        model.cuda()
        model.eval()

        pred, pred_tta = inference(test_loader, opt.tta)
        oof, oof_tta = inference(valid_loader, opt.tta)
        predictions = pred
        predictions_tta = pred_tta
        outoffolds = oof
        outoffolds_tta = oof_tta

        del model, valid_data; gc.collect()
        torch.cuda.empty_cache()

    np.save(export_dir/'outoffolds', outoffolds)
    np.save(export_dir/'predictions', predictions)
    if opt.tta:
        np.save(export_dir/'predictions_tta', predictions_tta)
        np.save(export_dir/'outoffolds_tta', outoffolds_tta)
    
    score = roc_auc_score(valid['target'].values, oof)
    score_tta =  roc_auc_score(valid['target'].values, oof_tta)
    LOGGER(f'score: {score:.5f}')
    LOGGER(f'score_tta: {score_tta:.5f}')
