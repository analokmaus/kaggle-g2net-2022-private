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

from datasets import G2Net2022Dataset, G2Net2022Dataset2
from architectures import *
from replknet import *
from models1d_pytorch import *
from loss_functions import BCEWithLogitsLoss, FocalLoss
from metrics import AUC
from transforms import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from training_extras import *


INPUT_DIR = Path('input/').expanduser()


class Baseline:
    name = 'baseline'
    seed = 2021
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/train_labels.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/train'
    test_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/sample_submission.csv'
    test_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/test'
    validate_on_train = False
    cv = 5
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    depth_bins = None # [0, 20, 40, 60, 80, 100, 1000]
    dataset = G2Net2022Dataset
    dataset_params = dict(
        normalize='local', 
        resize_factor=8, 
        spec_diff=True, match_time=False)

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
        EarlyStopping(patience=5, maximize=True, skip_epoch=4),
        SaveSnapshot()
    ]

    transforms = dict(
        train=A.Compose([ToTensorV2()]),
        test=A.Compose([ToTensorV2()]),
        tta=A.Compose([ToTensorV2()]),
    )

    pseudo_labels = None
    debug = False


class Baseline2(Baseline):
    name = 'aug_15'
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
    )
    batch_size = 32
    dataset_params = dict(
        normalize='local', 
        resize_factor=16, 
        spec_diff=True,
        match_time=False)
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            ToTensorV2(),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            TimeMaskingTensor(24, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v8.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v8/'


class Prep00(Baseline2):
    name = 'prep_00'
    dataset_params = dict(
        normalize='global', 
        resize_factor=16, 
        spec_diff=True,
        match_time=False)


class Aug00(Baseline2): # basic aug
    name = 'aug_00'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            # MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            ToTensorV2(),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            TimeMaskingTensor(24, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Aug01(Baseline2): # band noise
    name = 'aug_01'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25),
            RandomCrop(256),
            ToTensorV2(),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            TimeMaskingTensor(24, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Aug02(Baseline2): # delta noise
    name = 'aug_02'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25),
            DeltaNoise(strength=(5, 50), p=0.25),
            RandomCrop(256),
            ToTensorV2(),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            TimeMaskingTensor(24, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Aug02(Baseline2): # delta noise
    name = 'aug_02'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25),
            DeltaNoise(strength=(5, 50), p=0.25),
            RandomCrop(256),
            ToTensorV2(),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            TimeMaskingTensor(24, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Aug02prep0(Aug02):
    name = 'aug_02_prep0'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25),
            DeltaNoise(strength=(5, 50), p=0.25),
            RandomCrop(256),
            ClipSignal(0, 5),
            ToTensorV2(),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            TimeMaskingTensor(24, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ClipSignal(0, 5),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ClipSignal(0, 5),
            ToTensorV2()]),
    )


class Aug03(Baseline2): # delta noise + global
    name = 'aug_03'
    dataset_params = dict(
        normalize='global', 
        resize_factor=16, 
        spec_diff=True,
        match_time=False)
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25),
            DeltaNoise(strength=(1, 15), p=0.25),
            RandomCrop(256),
            ToTensorV2(),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            TimeMaskingTensor(24, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Aug03prep0(Aug03): # delta noise + global
    name = 'aug_03_prep0'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25), BandNoise(band_width=64, p=0.25),
            DeltaNoise(strength=(1, 15), p=0.25),
            RandomCrop(256),
            ClipSignal(0, 1),
            ToTensorV2(),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(24, p=0.5),
            TimeMaskingTensor(24, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ClipSignal(0, 1),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ClipSignal(0, 1),
            ToTensorV2()]),
    )
    

class Model00(Baseline2):
    name = 'model_00'
    dataset_params = dict(
        normalize='local', 
        resize_factor=16, 
        spec_diff=False, 
        match_time=False,
        random_crop=False)
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2)
    )

    
class Model02(Baseline2):
    name = 'model_02'
    model = create_RepLKNet31L
    model_params = dict(
        in_chans=3, num_classes=1
    )
    batch_size = 32
    weight_path = Path('input/RepLKNet-31L_ImageNet-22K.pth')


class Mixup00(Baseline2):
    name = 'mixup_00'
    hook = MixupTrain()


class Mixup01(Baseline2):
    name = 'mixup_01'
    hook = MixupTrain(alpha=2.0)


class Mixup02(Baseline2):
    name = 'mixup_03'
    hook = MixupTrain(lor_label=True)
