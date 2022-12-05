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

from datasets import *
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
    name = 'baseline_2'
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


class Baseline3(Baseline): # replicate 733
    name = 'baseline_3'
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
        # timm_params=dict(in_chans=2)
    )
    batch_size = 32
    dataset = G2Net2022Dataset3
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), NormalizeSpectrogram('mean'), AddChannel('diff')]),
        match_time=False)
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            ToTensorV2(),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.2),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v1.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v1/'


class Prep00(Baseline2): # global
    name = 'prep_00'
    dataset_params = dict(
        normalize='global', 
        resize_factor=16, 
        spec_diff=True,
        match_time=False)


class Prep01(Baseline2): # global 4ch
    name = 'prep_01'
    dataset = G2Net2022Dataset3
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), NormalizeSpectrogram('concat')
        ]),
        match_time=False)
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=4)
    )
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v7.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v7/'
    

class Prep02(Baseline2): # global 2ch
    name = 'prep_02'
    dataset = G2Net2022Dataset3
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), NormalizeSpectrogram('constant')
        ]),
        match_time=False)
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2)
    )
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v7.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v7/'


class Prep02ds0(Prep02):
    name = 'prep_02_ds0'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v9.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v9/'


class Prep03(Baseline3): # mean 2ch
    name = 'prep_03'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), NormalizeSpectrogram('mean')
        ]),
        match_time=False)
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2)
    )
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v10.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v10/'


class Ds00(Baseline2): # v7
    name = 'ds_00'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v7.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v7/'


class Ds01(Baseline3): # v9
    name = 'ds_01'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v9.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v9/'


class Ds01prep0(Ds01): # v9 global
    name = 'ds_01_prep0'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), 
            NormalizeSpectrogram('constant'), AddChannel('diff')
        ]),
        match_time=False)


class Ds02(Baseline3): # v10
    name = 'ds_02'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v10.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v10/'
    callbacks = [
        EarlyStopping(patience=10, maximize=True, skip_epoch=4),
        SaveSnapshot()
    ]


class Ds02prep0(Ds02): # v10 global
    name = 'ds_02_prep0'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), 
            NormalizeSpectrogram('constant'), AddChannel('diff')
        ]),
        match_time=False)


class Ds03(Baseline3): # v11
    name = 'ds_03'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v11.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v11/'
    callbacks = [
        EarlyStopping(patience=10, maximize=True, skip_epoch=4),
        SaveSnapshot()
    ]


class Ds03prep0(Ds03): # v11 global
    name = 'ds_03_prep0'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), 
            NormalizeSpectrogram('constant'), AddChannel('diff')
        ]),
        match_time=False)


class Ds04(Baseline3): # v12
    name = 'ds_04'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v12.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v12/'
    callbacks = [
        EarlyStopping(patience=10, maximize=True, skip_epoch=4),
        SaveSnapshot()
    ]


class Ds04prep0(Ds04): # v12 global
    name = 'ds_04_prep0'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), 
            NormalizeSpectrogram('constant'), AddChannel('diff')
        ]),
        match_time=False)


class Ds04prep1(Ds04): # v12 global 2ch
    name = 'ds_04_prep1'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), 
            NormalizeSpectrogram('constant')
        ]),
        match_time=False)
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2)
    )


class Ds04prep2(Ds04): # v12 global 4ch
    name = 'ds_04_prep2'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), 
            NormalizeSpectrogram('concat')
        ]),
        match_time=False)
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=4)
    )


class Ds05(Baseline3): # v13 global 2ch
    name = 'ds_05'
    splitter = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=Baseline.seed)
    depth_bins = [0, 20, 30, 40, 51, 1000]
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), 
            NormalizeSpectrogram('constant')
        ]),
        match_time=False)
    model_params = dict(
        model_name='tf_efficientnet_b6_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2)
    )
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v13.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v13/'
    callbacks = [
        EarlyStopping(patience=10, maximize=True, skip_epoch=4),
        SaveSnapshot()
    ]
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            # MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Ds05aug0(Ds05):
    name = 'ds_05_aug0'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            # MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            DropChannel(p=0.25),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Ds05aug1(Ds05):
    name = 'ds_05_aug1'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            DropChannel(p=0.25),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Ds05aug2(Ds05):
    name = 'ds_05_aug2'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            # MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            RandomAmplify(p=0.25),
            DropChannel(p=0.25),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Ds06(Ds05):
    name = 'ds_06'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v14.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v14/'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            RandomCrop(256),
            RandomAmplify(p=0.25),
            DropChannel(p=0.25),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Ds07(Ds06):
    name = 'ds_07'
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/concat_v13_v14.csv')
    train_dir = None


class Ds08(Ds06):
    name = 'ds_08'
    depth_bins = [0, 25, 30, 35, 41, 1000]
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v15.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v15/'


class Ds09(Ds06):
    name = 'ds_09'
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/concat_v13_v14_v15.csv')
    train_dir = None


class Ds10(Ds06):
    name = 'ds_10'
    depth_bins = [0, 20, 30, 40, 51, 1000]
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v16.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v16/'


class Ds11(Ds06):
    name = 'ds_11'
    depth_bins = [0, 30, 40, 51, 1000]
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v17.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v17/'


class Ds12(Ds06):
    name = 'ds_12'
    depth_bins = [0, 30, 40, 51, 1000]
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/concat_v16_v17.csv')
    train_dir = None


class Ds13(Ds06):
    name = 'ds_13'
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/concat_v13_v14_v15_v16_v17.csv')
    train_dir = None


class Res00(Ds09):
    name = 'res_00'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(8), NormalizeSpectrogram('constant')]),
        match_time=False)
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=256, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(512),
            ToTensorV2(),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.2),
            TimeMaskingTensor(64, p=0.5),
            TimeMaskingTensor(64, p=0.5),
            TimeMaskingTensor(64, p=0.2)]),
        test=A.Compose([
            CropImage(512),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(512),
            ToTensorV2()]),
    )


class Res01(Ds09):
    name = 'res_01'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(4), NormalizeSpectrogram('constant')]),
        match_time=False)
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=512, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(1024),
            ToTensorV2(),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.2),
            TimeMaskingTensor(128, p=0.5),
            TimeMaskingTensor(128, p=0.5),
            TimeMaskingTensor(128, p=0.2)]),
        test=A.Compose([
            CropImage(1024),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(1024),
            ToTensorV2()]),
    )
    # parallel = 'ddp'
    # batch_size = 128 # 16 per gpu
    # optimizer_params = dict(lr=1e-3, weight_decay=1e-6)
    batch_size = 16 # 16 per gpu
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)


class Res01ddp(Res01):
    name = 'res_01_ddp'
    clip_grad = None
    parallel = 'ddp'
    batch_size = 32 # 8 per gpu
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)


class Aug01(Ds04prep1): # test noise type1
    name = 'aug_01'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            InjectTimeNoise('input/timenoise_v0.pickle', resize_factor=16, p=0.25),
            ToTensorV2(),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.2),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Aug01ds0(Aug01):
    name = 'aug_01_ds0'
    train_path = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v10.csv'
    train_dir = INPUT_DIR/'g2net-detecting-continuous-gravitational-waves/v10/'


class Aug01prep0(Aug01):
    name = 'aug_01_prep0'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            InjectTimeNoise('input/timenoise_v0.pickle', resize_factor=16, p=0.25),
            ClipSignal(0, 1.0),
            ToTensorV2(),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.2),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ClipSignal(0, 1.0),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ClipSignal(0, 1.0),
            ToTensorV2()]),
    )


class Aug02(Ds04prep1): # time noise / anomaly
    name = 'aug_02'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=128, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            InjectTimeNoise('input/timenoise_v1.pickle', strength=(0.9, 1.5), resize_factor=16, p=0.25),
            InjectAnomaly('input/anomaly_v0_16.pickle', 'H1', p=0.05),
            InjectAnomaly('input/anomaly_v0_16.pickle', 'L1', p=0.05),
            ClipSignal(0, 3.0),
            ToTensorV2(),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.2),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ClipSignal(0, 3.0),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ClipSignal(0, 3.0),
            ToTensorV2()]),
    )


class Aug03(Ds04prep1):
    name = 'aug_03'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=0, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            InjectTimeNoise('input/timenoise_v1.pickle', strength=(0.9, 1.5), resize_factor=16, p=0.5),
            RandomAmplify(p=0.5),
            ToTensorV2(),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Aug04(Ds04prep1):
    name = 'aug_04'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=0, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            InjectTimeNoise('input/timenoise_v0.pickle', strength=(0.9, 1.5), resize_factor=16, p=0.5),
            RandomAmplify(p=0.5),
            DropChannel(p=0.25),
            ToTensorV2(),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.2),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Aug05(Ds04prep1):
    name = 'aug_05'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=0, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            InjectTimeNoise('input/timenoise_v2.pickle', strength=(0.9, 1.5), resize_factor=16, p=0.4),
            RandomAmplify(p=0.5),
            DropChannel(p=0.25),
            ToTensorV2(),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.2),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            InjectTimeNoise(
                'input/timenoise_v2.pickle', strength=(0.75, 1.0), resize_factor=16, p=0.1),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            ToTensorV2()]),
    )


class Aug06(Ds04prep1): # local + aug 04
    name = 'aug_06'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(16), 
            NormalizeSpectrogram('mean')
        ]),
        match_time=False)
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=0, y_max=180, p=0.5),
            MixupChannel(num_segments=20, fix_area=True, p=0.5),
            RandomCrop(256),
            InjectTimeNoise('input/timenoise_v0.pickle', strength=(0.9, 1.5), resize_factor=16, p=0.5),
            RandomAmplify(p=0.5),
            NormalizeSpectrogram('mean'),
            DropChannel(p=0.25),
            ToTensorV2(),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.5),
            FrequencyMaskingTensor(18, p=0.2),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.5),
            TimeMaskingTensor(32, p=0.2)]),
        test=A.Compose([
            CropImage(256),
            NormalizeSpectrogram('mean'),
            ToTensorV2()]),
        tta=A.Compose([
            CropImage(256),
            NormalizeSpectrogram('mean'),
            ToTensorV2()]),
    )


class Model00(Ds06):
    name = 'model_00'
    model = create_RepLKNet31L
    model_params = dict(
        in_chans=2, num_classes=1
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
