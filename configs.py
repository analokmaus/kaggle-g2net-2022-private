from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from transforms import FrequencyMaskingTensor, TimeMaskingTensor

from kuma_utils.torch.callbacks import (
    EarlyStopping, SaveSnapshot, SaveEveryEpoch, SaveAllSnapshots)
from kuma_utils.torch.hooks import TrainHook

from datasets import G2Net2022Dataset
from architectures import *
from replknet import create_RepLKNet31B
from models1d_pytorch import *
from loss_functions import BCEWithLogitsLoss
from metrics import AUC
from transforms import *


INPUT_DIR = Path('input/').expanduser()


class Baseline:
    name = 'baseline'
    seed = 2021
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v0.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v0'
    test_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/sample_submission.csv'
    test_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/test'
    cv = 5
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    dataset = G2Net2022Dataset
    dataset_params = dict(normalize='none', spec_diff=True, resize_factor=8)

    model = SimpleCNN
    model_params = dict(
        model_name='tf_efficientnet_b0',
        pretrained=True,
        num_classes=1,
        timm_params=dict(
            in_chans=3,
        )
    )
    weight_path = None
    num_epochs = 20
    batch_size = 128
    optimizer = optim.Adam
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    scheduler_target = None
    batch_scheduler = False
    criterion = nn.BCEWithLogitsLoss()
    eval_metric = AUC().torch
    monitor_metrics = []
    amp = True
    parallel = None
    deterministic = False
    clip_grad = 'value'
    max_grad_norm = 10000
    hook = TrainHook()
    callbacks = [
        EarlyStopping(patience=5, maximize=True),
        SaveSnapshot()
    ]

    transforms = dict(
        train=A.Compose([
            A.Normalize(), ToTensorV2()]),
        test=A.Compose([
            A.Normalize(), ToTensorV2()]),
        tta=A.Compose([
            A.Normalize(), ToTensorV2()]),
    )

    pseudo_labels = None
    debug = False


class Channel00(Baseline):
    name = 'channel_00'
    dataset_params = dict(normalize='local', resize_factor=8)
    model_params = dict(
        model_name='tf_efficientnet_b0',
        pretrained=True,
        num_classes=1,
        timm_params=dict(
            in_chans=2,
        )
    )
    transforms = dict(
        train=A.Compose([
            ToTensorV2()]),
        test=A.Compose([
            ToTensorV2()]),
        tta=A.Compose([
            ToTensorV2()]),
    )


class Aug00(Baseline):
    name = 'aug_00'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(), ToTensorV2(),
            FrequencyMaskingTensor(72, p=0.5),
            TimeMaskingTensor(128, p=0.5)]),
        test=A.Compose([
            A.Normalize(), ToTensorV2()]),
        tta=A.Compose([
            A.Normalize(), ToTensorV2()]),
    )


class Prep00(Aug00): # A1
    name = 'prep_00'
    dataset_params = dict(normalize='none', spec_diff=True, match_time=True, resize_factor=8)


class Leplk00(Aug00): # A1
    name = 'leplk_00'
    model = create_RepLKNet31B
    model_params = dict(
        in_chans=3, num_classes=1
    )
    batch_size = 64
