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

from kuma_utils.torch import TorchTrainer, TorchLogger
from kuma_utils.torch.utils import get_time, seed_everything, fit_state_dict

from configs import *
# from transforms import Compose, FlipWave
from utils import print_config, notify_me
from training_extras import make_tta_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='Baseline',
                        help="config name in configs.py")
    parser.add_argument("--num_workers", type=int, default=0)
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
        train = train.iloc[:10000]
        test = test.iloc[:1000]
    train = train.loc[train['target'] != -1]
    valid = pd.read_csv('input/g2net-detecting-continuous-gravitational-waves/train_labels.csv')
    valid = valid.loc[valid['target'] != -1]
    valid_dir = Path('input/g2net-detecting-continuous-gravitational-waves/train')
    
    '''
    Training
    '''
    scores = []
    train_data = cfg.dataset(
        df=train, data_dir=cfg.train_dir,
        transforms=cfg.transforms['train'], **cfg.dataset_params)
    valid_data = cfg.dataset(
        df=valid, data_dir = valid_dir,
        transforms=cfg.transforms['test'], **cfg.dataset_params)

    train_loader = D.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=False)
    valid_loader = D.DataLoader(
        valid_data, batch_size=cfg.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=False)
    fold = 0
    if opt.skip_existing and (export_dir/f'fold{fold}.pt').exists():
        LOGGER(f'checkpoint fold{fold}.pt already exists.')
    else:

        LOGGER(f'===== TRAINING FOLD {fold} =====')

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
    outoffolds = np.full((len(valid), 1), 0.5, dtype=np.float32)
    test_data = cfg.dataset(
        df=test, data_dir=cfg.test_dir,
        transforms=cfg.transforms['test'], **cfg.dataset_params)
    test_loader = D.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=opt.num_workers, pin_memory=False)
    fold = 0

    if not (export_dir/f'fold{fold}.pt').exists():
        LOGGER(f'fold{fold}.pt missing. No target to predict.')
        pass
    
    else:

        LOGGER(f'===== INFERENCE FOLD {fold} =====')

        model = cfg.model(**cfg.model_params)
        checkpoint = torch.load(export_dir/f'fold{fold}.pt', 'cpu')
        fit_state_dict(checkpoint['model'], model)
        model.load_state_dict(checkpoint['model'])
        scores.append(checkpoint['state']['best_score'])
        del checkpoint; gc.collect()
        if cfg.parallel == 'ddp':
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        trainer = TorchTrainer(model, serial=f'fold{fold}', device=None)
        trainer.register(hook=cfg.hook, callbacks=cfg.callbacks)

        if opt.tta: # flip wave TTA
            # tta_transform = Compose(
            #     cfg.transforms['test'].transforms + [FlipWave(always_apply=True)])
            # LOGGER(f'[{fold}] pred0 {test_loader.dataset.transforms}')
            # prediction0 = trainer.predict(test_loader, progress_bar=opt.progress_bar)
            # test_loader = make_tta_dataloader(test_loader, cfg.dataset, dict(
            #     paths=test['path'].values, transforms=tta_transform, 
            #     cache=test_cache, is_test=True, **cfg.dataset_params
            # ))
            # LOGGER(f'[{fold}] pred1 {test_loader.dataset.transforms}')
            # prediction1 = trainer.predict(test_loader, progress_bar=opt.progress_bar)
            # prediction_fold = (prediction0 + prediction1) / 2

            # LOGGER(f'[{fold}] oof0 {valid_loader.dataset.transforms}')
            # outoffold0 = trainer.predict(valid_loader, progress_bar=opt.progress_bar)
            # valid_loader = make_tta_dataloader(valid_loader, cfg.dataset, dict(
            #     paths=valid_fold['path'].values, targets=valid_fold['target'].values,
            #     cache=train_cache, transforms=tta_transform, is_test=True,
            #     **cfg.dataset_params))
            # LOGGER(f'[{fold}] oof1 {valid_loader.dataset.transforms}')
            # outoffold1 = trainer.predict(valid_loader, progress_bar=opt.progress_bar)
            # outoffold = (outoffold0 + outoffold1) / 2
            pass
        else:
            prediction_fold = trainer.predict(test_loader, progress_bar=opt.progress_bar)
            outoffold = trainer.predict(valid_loader, progress_bar=opt.progress_bar)

        predictions = prediction_fold
        outoffolds = outoffold

        del model, trainer; gc.collect()
        torch.cuda.empty_cache()

    if opt.tta:
        np.save(export_dir/'outoffolds_tta', outoffolds)
        np.save(export_dir/'predictions_tta', predictions)
    else:
        np.save(export_dir/'outoffolds', outoffolds)
        np.save(export_dir/'predictions', predictions)

    LOGGER(f'scores: {scores}')
    LOGGER(f'mean +- std: {np.mean(scores):.5f} +- {np.std(scores):.5f}')
