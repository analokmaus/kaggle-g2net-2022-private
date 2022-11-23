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
from loss_functions import BCEWithLogitsLoss
from metrics import AUC
from transforms import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


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
    dataset_params = dict(normalize='local', resize_factor=8, spec_diff=True, match_time=False)

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
        train=A.Compose([ToTensorV2()]),
        test=A.Compose([ToTensorV2()]),
        tta=A.Compose([ToTensorV2()]),
    )

    pseudo_labels = None
    debug = False


class Dataset00(Baseline):
    name = 'dataset_00'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v0.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v0/'


class Aug00(Dataset00):
    name = 'aug_00'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
            FrequencyMaskingTensor(72, p=0.5),
            TimeMaskingTensor(128, p=0.5)]),
        test=A.Compose([
            ToTensorV2()]),
        tta=A.Compose([
            ToTensorV2()]),
    )


class Model00(Aug00): # A1
    name = 'model_00'
    model = create_RepLKNet31L
    model_params = dict(
        in_chans=3, num_classes=1
    )
    batch_size = 32


class Model01(Aug00): # A1
    name = 'model_01'
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
    )
    batch_size = 32


class Model01ds0(Model01): # A1
    name = 'model_01_ds0'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/train_labels.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/train'


class Model01val0(Model01): # A1
    name = 'model_01_val0'
    splitter = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=Baseline.seed)
    depth_bins = [0, 20, 40, 60, 80, 100, 1000]


class Model01ds1(Model01val0):
    name = 'model_01_ds1'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v1.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v1/'
    

class Aug01(Model01ds1):
    name = 'aug_01'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            ToTensorV2(),
            FrequencyMaskingTensor(72, p=0.5),
            TimeMaskingTensor(128, p=0.5)]),
        test=A.Compose([
            ToTensorV2()]),
        tta=A.Compose([
            ToTensorV2()]),
    )


class Aug02(Aug01): # A2
    name = 'aug_02'
    dataset_params = dict(
        normalize='local', 
        resize_factor=8, 
        spec_diff=True, 
        match_time=False,
        random_crop=True)


class Aug02mod0(Aug02): # A2
    name = 'aug_02_mod0'
    dataset_params = dict(
        normalize='local', 
        resize_factor=8, 
        spec_diff=False, 
        match_time=False,
        random_crop=True)
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2)
    )


class Aug02ds0(Aug02): # A2
    name = 'aug_02_ds0'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v0.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v0/'


class Aug02ds1(Aug02): # A2
    name = 'aug_02_ds1'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/train_labels.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/train/'
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=Baseline.seed)
    depth_bins = None
    

class Aug03(Aug01):
    name = 'aug_03'
    dataset_params = dict(
        normalize='local', 
        resize_factor=8, 
        spec_diff=True, 
        match_time=False,
        random_crop=True,
        random_shift=True)


class Aug04(Aug02):
    name = 'aug_04'
    dataset_params = dict(
        normalize='local', 
        resize_factor=8, 
        spec_diff=True, 
        match_time=False,
        random_crop=False)
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            RandomCrop(512),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(36, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(36, p=0.5)]),
        test=A.Compose([
            CropImage(512),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(512),
            ToTensorV2()]),
    )


class Aug04mod0(Aug04):
    name = 'aug_04_mod0'
    dataset_params = dict(
        normalize='local', 
        resize_factor=8, 
        spec_diff=False, 
        match_time=False,
        random_crop=True)
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2)
    )


class Aug04ds0(Aug04):
    name = 'aug_04_ds0'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v0.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v0/'


class Aug04ds1(Aug04):
    name = 'aug_04_ds1'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v2.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v2/'


class Aug04ds2(Aug04):
    name = 'aug_04_ds2'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v1_cutoff_005.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v1/'


class Aug04ds3(Aug04):
    name = 'aug_04_ds3'
    dataset = G2Net2022Dataset2
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/concat_v0_v1_v2.csv'
    train_dir = None
    test_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/test.csv'
    test_dir = None


class Aug04prep0(Aug04):
    name = 'aug_04_prep0'
    dataset_params = dict(
        normalize='local', 
        resize_factor=8, 
        spec_diff=True, 
        match_time=True)


class Aug04prep1(Aug04):
    name = 'aug_04_prep1'
    dataset_params = dict(
        normalize='laeyoung', 
        resize_factor=8, 
        spec_diff=True, 
        match_time=True,
        fillna=True)


class Aug04prep2(Aug04):
    name = 'aug_04_prep2'
    dataset_params = dict(
        normalize='laeyoung', 
        resize_factor=10, 
        spec_diff=True, 
        match_time=True,
        fillna=True)
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            RandomCrop(512),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(36, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(36, p=0.5)]),
        test=A.Compose([
            CropImage(512),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(512),
            ToTensorV2()]),
    )


class Aug04mod1(Aug04):
    name = 'aug_04_mod1'
    model = create_RepLKNet31L
    model_params = dict(
        in_chans=3, num_classes=1
    )
    batch_size = 32
    weight_path = Path('input/RepLKNet-31L_ImageNet-22K.pth')


class Aug04mod2(Aug04prep2):
    name = 'aug_04_mod2'
    dataset_params = dict(
        normalize='laeyoung', 
        resize_factor=8, 
        spec_diff=False, 
        match_time=True,
        fillna=True)
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2)
    )


class Aug05(Aug04):
    name = 'aug_05'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            ToTensorV2(),
            FrequencyMaskingTensor(36, p=0.5),
            TimeMaskingTensor(48, p=0.5),
            FrequencyMaskingTensor(36, p=0.5),
            TimeMaskingTensor(48, p=0.5)]),
        test=A.Compose([
            ToTensorV2()]),
        tta=A.Compose([
            ToTensorV2()]),
    )


class Aug06(Aug04):
    name = 'aug_06'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(36, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(36, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(36, p=0.5)]),
        test=A.Compose([
            ToTensorV2()]),
        tta=A.Compose([
            ToTensorV2()]),
    )


class Aug06ds0(Aug06):
    name = 'aug_06_ds0'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v2.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v2/'


class Aug07(Aug04):
    name = 'aug_07'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            ToTensorV2(),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(12, p=0.5),
            TimeMaskingTensor(18, p=0.5)]),
        test=A.Compose([
            ToTensorV2()]),
        tta=A.Compose([
            ToTensorV2()]),
    )


    
class Model02(Aug02):
    name = 'model_02'
    model = create_RepLKNet31L
    model_params = dict(
        in_chans=3, num_classes=1
    )
    batch_size = 32
    weight_path = Path('input/RepLKNet-31L_ImageNet-22K.pth')
