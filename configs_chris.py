from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from transforms import FrequencyMaskingTensor, TimeMaskingTensor

from kuma_utils.torch.callbacks import (
    EarlyStopping, SaveSnapshot, SaveEveryEpoch, SaveAllSnapshots)
from kuma_utils.torch.hooks import TrainHook

from datasets import *
from architectures import *
from replknet import *
from models1d_pytorch import *
from loss_functions import BCEWithLogitsLoss, FocalLoss
from metrics import AUC
from transforms import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from training_extras import *

from team.chris_model import *


INPUT_DIR = Path('input/').expanduser()


class Chrisv16:
    name = 'chris_v16'
    seed = 2021
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/concat_v13_v14_v15.csv')
    train_dir = None
    valid_path = Path('input/g2net-detecting-continuous-gravitational-waves/v18v.csv')
    valid_dir = None
    test_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/sample_submission.csv'
    test_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/test'
    cv = 5
    splitter = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    depth_bins = [0, 20, 30, 40, 51, 1000]
    dataset = ChrisDataset
    dataset_params = dict(
        img_size=720,
        max_size=6000)

    model = Modelv16
    model_params = dict(
       name='tf_efficientnet_b7_ns',
       pretrained=True
    )
    weight_path = None
    num_epochs = 20
    batch_size = 64 # 16 per gpu
    optimizer = optim.AdamW
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    scheduler_target = None
    batch_scheduler = False
    criterion = BCEWithLogitsLoss()
    eval_metric = AUC().torch
    monitor_metrics = []
    amp = True
    parallel = 'ddp'
    deterministic = False
    clip_grad = 'value'
    max_grad_norm = 10000
    hook = ChrisTrain()
    callbacks = [
        EarlyStopping(patience=10, maximize=True, skip_epoch=4),
        SaveSnapshot()
    ]

    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(72, p=0.5),
            TimeMaskingTensor(72, p=0.5),
            TimeMaskingTensor(72, p=0.5)]),
        test=A.Compose([ToTensorV2()]),
        tta=A.Compose([ToTensorV2()]),
    )

    pseudo_labels = None
    debug = False


class C16val0(Chrisv16):
    name = 'chris_v16_val0'
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=Chrisv16.seed)
    depth_bins = None


class C16aug0(C16val0):
    name = 'chris_v16_aug0'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(360, 180, p=0.5),
            DropChannel(p=0.25),
            ClipSignal(-20, 20),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(72, p=0.5),
            TimeMaskingTensor(72, p=0.5),
            TimeMaskingTensor(72, p=0.5)]),
        test=A.Compose([ClipSignal(-20, 20), ToTensorV2()]),
        tta=A.Compose([ClipSignal(-20, 20), ToTensorV2()]),
    )


class C16aug1(C16val0):
    name = 'chris_v16_aug1'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(360, 180, p=0.5),
            RandomAmplify(p=0.5),
            ClipSignal(-20, 20),
            DropChannel(p=0.25),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(72, p=0.5),
            TimeMaskingTensor(72, p=0.5),
            TimeMaskingTensor(72, p=0.5)]),
        test=A.Compose([ClipSignal(-20, 20), ToTensorV2()]),
        tta=A.Compose([ClipSignal(-20, 20), ToTensorV2()]),
    )
