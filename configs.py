from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
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
from loss_functions import *
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
    loader_to_callback = False
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


class Baseline1d(Baseline3):
    name = 'baseline1d'
    model = ResNet1d
    model_params = dict(
        in_channels=2, 
        base_filters=64,
        kernel_size=16, 
        stride=2, 
        groups=64, 
        n_block=16, 
        n_classes=1,
        use_bn=True,
        dropout=0.2
    )
    dataset_params = dict(
        preprocess=A.Compose([ToWaveform()])
    )
    transforms = dict(
        train=A.Compose([
            RandomCrop(4096),
            WaveToTensor()]),
        test=A.Compose([
            CropImage(4096),
            WaveToTensor()]),
        tta=A.Compose([
            CropImage(4096),
            WaveToTensor()]),
    )
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/concat_v13_v14_v15.csv')
    train_dir = None
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)


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
            CropImage(256), ToTensorV2(), RemoveAnomaly()]),
    )


class Ds09(Ds06):
    name = 'ds_09'
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/concat_v13_v14_v15.csv')
    train_dir = None
    # valid_path = Path('input/g2net-detecting-continuous-gravitational-waves/v18v.csv')
    valid_path = Path('input/g2net-detecting-continuous-gravitational-waves/v21v.csv')
    valid_dir = None


class Ds09val0(Ds09):
    name = 'ds_09_val0'
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=Baseline.seed)
    depth_bins = None
    parallel = 'ddp'
    batch_size = 64 # 32 per gpu
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)


class Ds09prep0(Ds09): # cf. Aug03
    name = 'ds_09_prep0'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(img_size=720), 
            NormalizeSpectrogram('chris')
        ]),
        match_time=False)
    parallel = 'ddp'
    batch_size = 32
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=360, y_max=180, p=0.5),
            RandomAmplify(p=0.25),
            DropChannel(p=0.25),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),]),
        test=A.Compose([
            ToTensorV2()]),
        tta=A.Compose([
            ToTensorV2(), RemoveAnomaly()]),
    )


class Ds09mod6(Ds09prep0):
    name = 'ds_09_mod6'
    model_params = dict(
        model_name='tf_efficientnet_b7_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2),
        custom_preprocess='chris_debias',
        custom_classifier='avg'
    )


class Ds09val1(Ds09mod6):
    name = 'ds_09_val1'
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=Baseline.seed)
    depth_bins = None


class Ds09mod7(Ds09val1):
    name = 'ds_09_mod7'
    model_params = dict(
        model_name='tf_efficientnet_b7_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2),
        custom_preprocess='chris_debias',
        custom_classifier='gem',
        custom_attention='triplet'
    )


class Ds09mod8(Ds09val1):
    name = 'ds_09_mod8'
    splitter = Path('results/ds_09_val1/folds.pickle')
    model_params = dict(
        model_name='inception_v4',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2),
        custom_preprocess='chris_debias',
        custom_classifier='gem',
    )


class Ds14(Ds09mod6):
    name = 'ds_14' # full data training
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/concat_v18n1_v18n2_v19_v20.csv')
    model = ClassificationModel
    model_params = dict(
        classification_model='tf_efficientnet_b7_ns',
        # classification_model='swin_base_patch4_window12_384',
        in_chans=2,
        num_classes=1,
        custom_preprocess='chris_debias',
        custom_classifier='avg',
        pretrained=True
    )
    batch_size = 32
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=360, y_max=180, p=0.5),
            RandomAmplify(p=0.25),
            DropChannel(p=0.25),
            ToTensorV2(),
            RemoveAnomaly(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),]),
        test=A.Compose([
            ToTensorV2(), RemoveAnomaly()]),
        tta=A.Compose([
            ToTensorV2(), RemoveAnomaly()]),
    )


class Ds14prep0(Ds14):
    name = 'ds_14_prep0'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(img_size=720), 
            NormalizeSpectrogram('column_wise')
        ]),
        match_time=False)
    num_epochs = 25
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ShiftImage(x_max=0, y_max=180, p=0.5),
            RandomAmplify(p=0.25),
            ClipSignal(-5, 5),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),]),
        test=A.Compose([
            ClipSignal(-5, 5), ToTensorV2()]),
        tta=A.Compose([
            ClipSignal(-5, 5), ToTensorV2()]),
    )


class Ds14prep1(Ds14prep0):
    name = 'ds_14_prep1'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(img_size=720), 
            NormalizeSpectrogram('column_row_wise')
        ]),
        match_time=False)

    
class Ds14prep2(Ds14):
    name = 'ds_14_prep2'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(img_size=720), 
            NormalizeSpectrogram('chris')
        ]),
        match_time=True)
    valid_path = Path('input/g2net-detecting-continuous-gravitational-waves/v23v.csv')


class Ds14prep3(Ds14):
    name = 'ds_14_prep3'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(img_size=720), 
            NormalizeSpectrogram('column_wise')
        ]),
        match_time=True)


class Ds15(Baseline3):
    name = 'ds_15'
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/concat_v18n1_v18n2.csv')
    train_dir = None
    valid_path = Path('input/g2net-detecting-continuous-gravitational-waves/v21v.csv')
    valid_dir = None
    dataset = G2Net2022Dataset8
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('chris')]),
        match_time=False,
        return_mask=False,
        positive_p=2/3,
        signal_path=Path('input/g2net-detecting-continuous-gravitational-waves/v18s.csv'),
        signal_dir=Path('input/g2net-detecting-continuous-gravitational-waves/v18s'),
        )
    model_params = dict(
        model_name='tf_efficientnet_b7_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2),
        custom_preprocess='chris_debias',
        custom_classifier='avg'
    )
    parallel = 'ddp'
    batch_size = 64 # 16 per gpu
    num_epochs = 50
    optimizer = optim.Adam
    optimizer_params = dict(lr=5e-4, weight_decay=1e-6)
    # scheduler = CosineAnnealingLR
    # scheduler_params = dict(T_max=5, eta_min=1e-6)
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=10, T_mult=1, eta_min=1e-6)
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            RandomAmplify(p=0.25),
            DropChannel(p=0.25),
            ToTensorV2(),
            RemoveAnomaly(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),]),
        test=A.Compose([
            ToTensorV2(), RemoveAnomaly()]),
        tta=A.Compose([
            ToTensorV2(), RemoveAnomaly()]),
    )
    callbacks = [
        EarlyStopping(patience=10, maximize=True, skip_epoch=4),
        SaveSnapshot()
    ]


class Ds15l(Ds15):
    name = 'ds_15_l'
    num_epochs = 100
    callbacks = [
        EarlyStopping(patience=30, maximize=True, skip_epoch=10),
        SaveSnapshot()
    ]


class Ds15aug0(Ds15):
    name = 'ds_15_aug0'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            RandomAmplify(p=0.25),
            MixupChannel(20, p=0.25),
            DropChannel(p=0.25),
            ToTensorV2(),
            RemoveAnomaly(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),]),
        test=A.Compose([
            ToTensorV2(), RemoveAnomaly()]),
        tta=A.Compose([
            ToTensorV2(), RemoveAnomaly()]),
    )


class Ds15aug1(Ds15):
    name = 'ds_15_aug1'
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('chris')]),
        match_time=False,
        return_mask=False,
        positive_p=2/3,
        signal_amplifier=1.5,
        noise_mixup_p=0.25,
        signal_path=Path('input/g2net-detecting-continuous-gravitational-waves/v18s.csv'),
        signal_dir=Path('input/g2net-detecting-continuous-gravitational-waves/v18s'),
        )


class Ds16(Ds15): # signal based sampling
    name = 'ds_16'
    dataset = G2Net2022Dataset88
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/v18s.csv')
    train_dir = Path('input/g2net-detecting-continuous-gravitational-waves/v18s')
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('chris')]),
        match_time=False,
        return_mask=False,
        positive_p=2/3,
        signal_amplifier=1.0,
        noise_mixup_p=0,
        noise_path=Path('input/g2net-detecting-continuous-gravitational-waves/concat_v18n1_v18n2.csv'),
        noise_dir=None,)
    num_epochs = 40
    callbacks = [
        EarlyStopping(patience=20, maximize=True, skip_epoch=10),
        SaveSnapshot()
    ]


class Ds16aug0(Ds16):
    name = 'ds_16_aug0'
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('chris')]),
        match_time=False,
        return_mask=False,
        positive_p=2/3,
        signal_amplifier=1.0,
        shift_range=(-165, 165),
        noise_mixup_p=0,
        noise_path=Path('input/g2net-detecting-continuous-gravitational-waves/concat_v18n1_v18n2.csv'),
        noise_dir=None,)


class Ds16aug1(Ds16):
    name = 'ds_16_aug1'
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('chris')]),
        match_time=False,
        return_mask=False,
        positive_p=2/3,
        signal_amplifier=1.0,
        shift_range=(-165, 165),
        noise_mixup_p=0.25,
        noise_path=Path('input/g2net-detecting-continuous-gravitational-waves/concat_v18n1_v18n2.csv'),
        noise_dir=None,)
    


class Ds16aug2(Ds16): 
    name = 'ds_16_aug2'
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('chris')]),
        match_time=False,
        return_mask=False,
        positive_p=2/3,
        signal_amplifier=np.linspace(2.0, 1.0, 40),
        noise_mixup_p=0,
        noise_path=Path('input/g2net-detecting-continuous-gravitational-waves/concat_v18n1_v18n2.csv'),
        noise_dir=None,)
    num_epochs = 40
    callbacks = [
        EarlyStopping(patience=20, maximize=True, skip_epoch=10),
        SaveSnapshot(), StepDataset()
    ]
    loader_to_callback = True


class Ds16aug3(Ds16): 
    name = 'ds_16_aug3'
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('chris')]),
        match_time=False,
        return_mask=False,
        positive_p=2/3,
        signal_amplifier=np.linspace(2.0, 1.0, 10),
        noise_mixup_p=0,
        noise_path=Path('input/g2net-detecting-continuous-gravitational-waves/concat_v18n1_v18n2.csv'),
        noise_dir=None,)
    num_epochs = 40
    callbacks = [
        EarlyStopping(patience=20, maximize=True, skip_epoch=10),
        SaveSnapshot(), StepDataset()
    ]
    loader_to_callback = True
    

class Aug04(Ds09val1):
    name = 'aug_04'
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(img_size=720), 
            NormalizeSpectrogram('constant')
        ]),
        match_time=False)
    

class Aug05(Ds09val1):
    name = 'aug_05'
    splitter = Path('results/ds_09_val1/folds.pickle')
    dataset_params = dict(
        preprocess=A.Compose([
            ToSpectrogram(), AdaptiveResize(img_size=1440), 
            NormalizeSpectrogram('chris')
        ]),
        match_time=False)


class Ds17(Ds15): # signal based sampling with chi 2 based noise
    name = 'ds_17'
    dataset = G2Net2022Dataset888
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/v18s.csv')
    train_dir = Path('input/g2net-detecting-continuous-gravitational-waves/v18s')
    valid_path = Path('input/g2net-detecting-continuous-gravitational-waves/v21v.csv')
    valid_dir = None
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('chris')]),
        match_time=False,
        return_mask=False,
        positive_p=2/3,
        signal_amplifier=1.0,
        shift_range=(-120, 120),
        test_stat=Path('input/signal_stat.pickle'),
        test_dir=Path('input/g2net-detecting-continuous-gravitational-waves/test/')
       )
    num_epochs = 40
    callbacks = [
        EarlyStopping(patience=20, maximize=True, skip_epoch=10),
        SaveSnapshot()
    ]
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            RandomAmplify(p=0.25),
            ClipSignal(-5, 5),
            ToTensorV2(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),]),
        test=A.Compose([
            ClipSignal(-5, 5), ToTensorV2()]),
        tta=A.Compose([
            ClipSignal(-5, 5), ToTensorV2()]),
    )
    model = ClassificationModel
    model_params = dict(
        classification_model='tf_efficientnet_b7_ns',
        in_chans=2,
        num_classes=1,
        custom_preprocess='chris_debias',
        custom_classifier='avg',
        pretrained=True
    )


class Ds17prep0(Ds17):
    name = 'ds_17_prep0'
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('chris')]),
        match_time=True,
        return_mask=False,
        positive_p=2/3,
        signal_amplifier=1.0,
        shift_range=(-120, 120),
        test_stat=Path('input/signal_stat.pickle'),
        test_dir=Path('input/g2net-detecting-continuous-gravitational-waves/test/')
       )


class Ds17prep1(Ds17):
    name = 'ds_17_prep1'
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('column_wise')]),
        match_time=True,
        return_mask=False,
        positive_p=2/3,
        signal_amplifier=1.0,
        shift_range=(-120, 120),
        test_stat=Path('input/signal_stat.pickle'),
        test_dir=Path('input/g2net-detecting-continuous-gravitational-waves/test/')
       )


class Ds17prep2(Ds17):
    name = 'ds_17_prep2'
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('column_wise')]),
        match_time=True,
        return_mask=False,
        positive_p=2/3,
        signal_amplifier=np.linspace(2.0, 1.0, 10),
        shift_range=(-120, 120),
        test_stat=Path('input/signal_stat.pickle'),
        test_dir=Path('input/g2net-detecting-continuous-gravitational-waves/test/')
       )
    callbacks = [
        EarlyStopping(patience=20, maximize=True, skip_epoch=10),
        SaveSnapshot(), StepDataset()
    ]
    loader_to_callback = True
    valid_path = Path('input/g2net-detecting-continuous-gravitational-waves/v23v.csv')


class Ds18(Ds17): # signal based sampling with chi 2 based noise Re/Im blending
    name = 'ds_17'
    dataset = G2Net2022Dataset8888
    train_path = Path('input/g2net-detecting-continuous-gravitational-waves/v18s.csv')
    train_dir = Path('input/g2net-detecting-continuous-gravitational-waves/v18s')
    valid_path = Path('input/g2net-detecting-continuous-gravitational-waves/v23v.csv')


class Ds18prep0(Ds18):
    name = 'ds_18_prep0'
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=720), NormalizeSpectrogram('column_wise')]),
        match_time=True,
        return_mask=False,
        positive_p=2/3,
        signal_amplifier=np.linspace(2.0, 1.0, 10),
        shift_range=(-120, 120),
        test_stat=Path('input/signal_stat.pickle'),
        test_dir=Path('input/g2net-detecting-continuous-gravitational-waves/test/')
       )
    callbacks = [
        EarlyStopping(patience=20, maximize=True, skip_epoch=10),
        SaveSnapshot(), StepDataset()
    ]
    loader_to_callback = True


class Mixup03(Ds09val1):
    name = 'mixup_03'
    hook = MixupTrain(alpha=4.0, lor_label=True)
    model_params = dict(
        model_name='tf_efficientnet_b7_ns',
        pretrained=True,
        num_classes=1,
        timm_params=dict(in_chans=2),
        custom_preprocess='chris_debias',
        custom_classifier='gem',
        custom_attention='mixup' # manifold mixup
    )


class Model01(Ds16aug0):
    name = 'model_01'
    dataset_params = dict(
        preprocess=A.Compose([
            AdaptiveResize(img_size=640), NormalizeSpectrogram('chris'), Crop352()]),
        match_time=False,
        return_mask=True,
        positive_p=2/3,
        signal_amplifier=1.0,
        shift_range=(-150, 150),
        noise_mixup_p=0.25,
        noise_path=Path('input/g2net-detecting-continuous-gravitational-waves/concat_v18n1_v18n2.csv'),
        noise_dir=None,)
    model = SegmentationAndClassification
    model_params = dict(
        segmentation_model='timm-efficientnet-b7',
        classification_model='tf_efficientnet_b0_ns',
        in_chans=2,
        num_classes=1,
        custom_preprocess='chris_debias',
        custom_classifier='avg',
        return_mask=True,
        pretrained=True
    )
    hook = SegAndClsTrain()
    criterion = BCEWithLogitsAux(weight=(0.6, 0.4))
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            RandomAmplify(p=0.25),
            DropChannel(p=0.25),
            ToTensorV2(transpose_mask=True),
            RemoveAnomaly(),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),]),
        test=A.Compose([
            ToTensorV2(transpose_mask=True), RemoveAnomaly()]),
        tta=A.Compose([
            ToTensorV2(transpose_mask=True), RemoveAnomaly()]),
    )


class Model01lf0(Model01):
    name = 'model_01_lf0'
    criterion = BCEWithLogitsAux(weight=(0.7, 0.3))


class Model02(Model01):
    name = 'model_02'
    model_params = dict(
        segmentation_model='timm-efficientnet-b7',
        classification_model='tf_efficientnet_b0_ns',
        in_chans=2,
        num_classes=1,
        concat_original=True,
        custom_preprocess='chris_debias',
        custom_classifier='avg',
        return_mask=True,
        pretrained=True
    )
    valid_path = Path('input/g2net-detecting-continuous-gravitational-waves/v21v.csv')


class Model02lf0(Model02):
    name = 'model_02_lf0'
    criterion = BCEWithLogitsAux(weight=(0.7, 0.3))


class Model02lf1(Model02):
    name = 'model_02_lf1'
    criterion = BCEWithLogitsAux(weight=(0.75, 0.25))


class Model02aug0(Model02):
    name = 'model_02_aug0'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            RandomAmplify(p=0.25),
            ToTensorV2(transpose_mask=True),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            FrequencyMaskingTensor(24, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),
            TimeMaskingTensor(96, p=0.5),]),
        test=A.Compose([
            ToTensorV2(transpose_mask=True)]),
        tta=A.Compose([
            ToTensorV2(transpose_mask=True)]),
    )


class Model02aug1(Model02):
    name = 'model_02_aug1'
    transforms = dict(
        train=A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            RandomAmplify(p=0.25),
            DropChannel(p=0.25),
            ToTensorV2(transpose_mask=True),
            RemoveAnomaly(),
            ]),
        test=A.Compose([
            ToTensorV2(transpose_mask=True), RemoveAnomaly()]),
        tta=A.Compose([
            ToTensorV2(transpose_mask=True), RemoveAnomaly()]),
    )

    