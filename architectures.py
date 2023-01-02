import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import segmentation_models_pytorch as smp
from models1d_pytorch import ResNet1D, CNN1d
from modules import *
from ffc_fix import FFC_BN_ACT

from kuma_utils.torch.modules import AdaptiveConcatPool2d, GeM, AdaptiveGeM
from kuma_utils.torch.utils import freeze_module


'''
Common
'''
def get_preprocess(name):
    if name == 'chris_debias':
        preprocess = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(3,31), stride=(1,2), padding=(3//2,31//2)),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1,2), padding=(5//2,5//2)),
            nn.GELU(),
        )
        out_chans = 128
    elif name == 'debias_large':
        preprocess = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(7, 39), stride=(1,2), padding=(15//2,31//2)),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1,2), padding=(5//2,5//2)),
            nn.GELU(),
        )
        out_chans = 128
    elif name == 'debias_small':
        preprocess = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(7, 21), stride=(1, 2), padding=(7//2, 21//2)),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1,2), padding=(5//2,5//2)),
            nn.GELU(),
        )
        out_chans = 128
    elif name == 'debias_raw_65':
        preprocess = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(7, 65), stride=(1,2), padding=(7//2, 65//2)),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(1,2), padding=(7//2, 7//2)),
            nn.GELU(),
        )
        out_chans = 128
    elif name == 'debias_raw_65_str4':
        preprocess = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(7, 65), stride=(1,4), padding=(7//2, 65//2)),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(1,2), padding=(7//2, 7//2)),
            nn.GELU(),
        )
        out_chans = 128
    elif name == 'debias_raw_65_str8':
        preprocess = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(7, 65), stride=(1,8), padding=(7//2, 65//2)),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(1,2), padding=(7//2, 7//2)),
            nn.GELU(),
        )
        out_chans = 128
    elif name == 'debias_3x31_2':
        preprocess = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(3,31), stride=(1,2), padding=(3//2,31//2)),
            nn.GELU(),
        )
        out_chans = 64
    elif name == 'ffc_15':
        preprocess = nn.Sequential(
            FFC_BN_ACT(
                2, 64, kernel_size=15, stride=1, padding=15//2, ratio_gin=0.5, ratio_gout=0.5),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 2), padding=(7//2, 7//2)),
            nn.GELU(),
        )
        out_chans = 128
    else:
        preprocess = nn.Identity()
        out_chans = 2
    
    return preprocess, out_chans


'''
New
'''
class SimpleCNN(nn.Module): # do NOT use

    def __init__(self, 
                 model_name='tf_efficientnet_b0', 
                 pretrained=False, 
                 num_classes=1,
                 timm_params={}, 
                 custom_preprocess='none',
                 custom_classifier='none',
                 custom_attention='none', 
                 dropout=0,
                 augmentations=None,
                 augmentations_test=None, 
                 resize_img=None,
                 upsample='nearest', 
                 mixup='mixup',
                 norm_spec=False,
                 return_spec=False):
        
        super().__init__()
        self.cnn = timm.create_model(model_name, 
                                     pretrained=pretrained, 
                                     num_classes=num_classes,
                                     **timm_params)
        self.mixup_mode = 'input'
        if custom_classifier != 'none' or custom_attention != 'none' or custom_preprocess != 'none':
            model_type = self.cnn.__class__.__name__
            try:
                feature_dim = self.cnn.get_classifier().in_features
                self.cnn.reset_classifier(0, '')
            except:
                raise ValueError(f'Unsupported model type: {model_type}')

            if custom_preprocess == 'laeyoung_debias':
                C, _, H, W = self.cnn.conv_stem.weight.shape
                preprocess = LargeKernel_debias(1, C, [H, W], 1, [H//2, W//2], 1, 1, False)
            elif custom_preprocess == 'chris_debias':
                preprocess = nn.Sequential(
                    nn.Conv2d(2, 64, kernel_size=(3,31), stride=(1,2), padding=(3//2,31//2)),
                    nn.GELU(),
                    nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1,2), padding=(5//2,5//2)),
                    nn.GELU(),
                )
                timm_params2 = timm_params.copy()
                timm_params2.update({
                    'in_chans': 128,
                    'global_pool': '',
                    'num_classes': 0
                })
                self.cnn = timm.create_model(model_name, 
                    pretrained=pretrained, **timm_params2)
            else:
                preprocess = nn.Identity()

            if custom_attention == 'triplet':
                attention = TripletAttention()
            elif custom_attention == 'mixup':
                attention = ManifoldMixup()
                self.mixup_mode = 'manifold'
            else:
                attention = nn.Identity()
            
            if custom_classifier == 'avg':
                global_pool = nn.AdaptiveAvgPool2d((1, 1))
            elif custom_classifier == 'max':
                global_pool = nn.AdaptiveMaxPool2d((1, 1))
            elif custom_classifier == 'concat':
                global_pool = AdaptiveConcatPool2d()
                feature_dim = feature_dim * 2
            elif custom_classifier == 'gem':
                global_pool = GeM(p=3, eps=1e-4)
            else:
                global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            self.cnn = nn.Sequential(
                preprocess, 
                self.cnn, 
                attention, 
                global_pool, 
                Flatten(),
                nn.Linear(feature_dim, 512), 
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.ReLU(inplace=True), 
                nn.Linear(512, num_classes)
            )
        self.norm_spec = norm_spec
        if self.norm_spec:
            self.norm = nn.BatchNorm2d(3)
        self.resize_img = resize_img
        if isinstance(self.resize_img, int):
            self.resize_img = (self.resize_img, self.resize_img)
        self.return_spec = return_spec
        self.augmentations = augmentations
        self.augmentations_test = augmentations_test
        self.upsample = upsample
        self.mixup = mixup
        assert self.mixup in ['mixup', 'cutmix']

    def feature_mode(self):
        self.cnn[-1] = nn.Identity()
        self.cnn[-2] = nn.Identity()

    def forward(self, spec, lam=None, idx=None): # spec: (batch size, ch, f, t)
        if lam is not None: # in-batch mixup
            if self.mixup == 'mixup' and self.mixup_mode == 'input':
                spec, lam = mixup(spec, lam, idx)
            elif self.mixup == 'cutmix':
                spec, lam = cutmix(spec, lam, idx)
        
        if self.training and self.augmentations is not None:
            spec = self.augmentations(spec)
        if not self.training and self.augmentations_test is not None:
            spec = self.augmentations_test(spec)

        if self.resize_img is not None:
            spec = F.interpolate(spec, size=self.resize_img, mode=self.upsample)

        if self.norm_spec:
            spec = self.norm(spec)
        
        if self.mixup_mode == 'manifold':
            self.cnn[2].update(lam, idx)

        if self.return_spec and lam is not None:
            return self.cnn(spec), spec, lam
        elif self.return_spec:
            return self.cnn(spec), spec
        elif lam is not None:
            return self.cnn(spec), lam
        else:
            return self.cnn(spec)


class ClassificationModel(nn.Module):

    def __init__(self,
                 classification_model='resnet18',
                 classification_params={},
                 in_chans=2,
                 num_classes=1,
                 custom_preprocess='none',
                 custom_classifier='none',
                 custom_attention='none', 
                 dropout=0,
                 pretrained=False):

        super().__init__()

        self.preprocess, cls_in_chans = get_preprocess(custom_preprocess)
        
        self.classification_model = timm.create_model(
            classification_model,
            pretrained=pretrained,
            in_chans=cls_in_chans,
            num_classes=num_classes,
            **classification_params
        )
        feature_dim = self.classification_model.get_classifier().in_features
        self.classification_model.reset_classifier(0, '')

        if custom_attention == 'triplet':
            self.attention = TripletAttention()
        else:
            self.attention = nn.Identity()

        if custom_classifier == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif custom_classifier == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif custom_classifier == 'concat':
            self.global_pool = AdaptiveConcatPool2d()
            feature_dim = feature_dim * 2
        elif custom_classifier == 'gem':
            self.global_pool = GeM(p=3, eps=1e-4)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if 'Transformer' in self.classification_model.__class__.__name__:
            self.classification_model.patch_embed = CustomHybdridEmbed(
                self.classification_model.patch_embed.proj, 
                channel_in=cls_in_chans,
                transformer_original_input_size=(1, cls_in_chans, *self.classification_model.patch_embed.img_size)
            )
            self.is_tranformer = True
        else:
            self.is_tranformer = False

        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(feature_dim, 512), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.ReLU(inplace=True), 
            nn.Linear(512, num_classes))

    def forward(self, x):
        output = self.classification_model(self.preprocess(x))
        output = self.attention(output)
        if self.is_tranformer:
            output = output.mean(dim=1)
        else:
            output = self.global_pool(output)
        output = self.head(output)
        return output


class SegmentationAndClassification(nn.Module):

    def __init__(self,
                 segmentation_model='se_resnext50_32x4d',
                 segmentation_params={},
                 classification_model='resnet18',
                 classification_params={},
                 in_chans=2,
                 num_classes=1,
                 custom_preprocess='none',
                 custom_classifier='none',
                 custom_attention='none', 
                 dropout=0,
                 pretrained=False,
                 return_mask=False,
                 concat_original=False):

        super().__init__()
        self.return_mask = return_mask
        self.concat = concat_original

        self.preprocess, seg_in_chans = get_preprocess(custom_preprocess)

        self.segmentation_model = smp.Unet(
            encoder_name=segmentation_model,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=seg_in_chans,
            classes=1,
            **segmentation_params
        )

        self.classification_model = timm.create_model(
            classification_model,
            pretrained=pretrained,
            in_chans=in_chans+1 if self.concat else 1,
            num_classes=num_classes,
            **classification_params
        )
        feature_dim = self.classification_model.get_classifier().in_features
        self.classification_model.reset_classifier(0, '')

        if custom_attention == 'triplet':
            self.attention = TripletAttention()
        else:
            self.attention = nn.Identity()

        if custom_classifier == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif custom_classifier == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif custom_classifier == 'concat':
            self.global_pool = AdaptiveConcatPool2d()
            feature_dim = feature_dim * 2
        elif custom_classifier == 'gem':
            self.global_pool = GeM(p=3, eps=1e-4)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(feature_dim, 512), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.ReLU(inplace=True), 
            nn.Linear(512, num_classes))

    def forward(self, x):
        mask = self.segmentation_model(self.preprocess(x))

        if self.concat:
            if mask.shape[3] != x.shape[3]:
                mask =  F.interpolate(mask, size=(mask.shape[2], x.shape[3]))
            output = torch.cat([x, mask], axis=1)
        else:
            output = mask

        output = self.classification_model(output)
        output = self.attention(output)
        output = self.global_pool(output)
        output = self.head(output)

        if self.return_mask:
            return output, mask
        else:
            return output


class MatchedFilterModel(nn.Module):

    def __init__(self,
            wave_bank_path,
            filter_height=90,
            resize_factor=1,
            num_filters=1024,
            filter_method='sum',
            classification_model='resnet18',
            classification_params={},
            in_chans=2,
            num_classes=1,
            custom_classifier='none',
            custom_attention='none', 
            dropout=0,
            pretrained=False):

        super().__init__()
        
        wave_bank = torch.load(wave_bank_path, 'cpu')
        wave_bank = wave_bank[:, 180-filter_height//2:180+filter_height//2+1, :]
        num_filters_0, filter_height, filter_width = wave_bank.shape
        if filter_method == 'sum':
            wave_bank = wave_bank.reshape(num_filters, num_filters_0//num_filters, filter_height, filter_width)
            wave_bank = wave_bank.sum(1)
        else:
            wave_bank = wave_bank[:num_filters]
        wave_bank = torch.stack([wave_bank, wave_bank], dim=1) # (num_filters, 2, filter_height, filter_width)
        self.filter = torch.nn.Conv2d(
            in_chans, num_filters, kernel_size=(filter_height, filter_width), stride=1, padding=(filter_height//2, 0), bias=False)
        self.filter.weight = nn.Parameter(wave_bank)
        self.freeze_filter()
        self.resize_f = resize_factor
        del wave_bank
        
        self.classification_model = timm.create_model(
            classification_model,
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
            **classification_params
        )
        feature_dim = self.classification_model.get_classifier().in_features
        self.classification_model.reset_classifier(0, '')

        if custom_attention == 'triplet':
            self.attention = TripletAttention()
        else:
            self.attention = nn.Identity()

        if custom_classifier == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif custom_classifier == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif custom_classifier == 'concat':
            self.global_pool = AdaptiveConcatPool2d()
            feature_dim = feature_dim * 2
        elif custom_classifier == 'gem':
            self.global_pool = GeM(p=3, eps=1e-4)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if 'Transformer' in self.classification_model.__class__.__name__:
            self.classification_model.patch_embed = CustomHybdridEmbed(
                self.classification_model.patch_embed.proj, 
                channel_in=1,
                transformer_original_input_size=(1, 1, *self.classification_model.patch_embed.img_size)
            )
            self.is_tranformer = True
        else:
            self.is_tranformer = False

        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(feature_dim, 512), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.ReLU(inplace=True), 
            nn.Linear(512, num_classes))

    def forward(self, x):
        output = self.filter(x) # (bs, num_filters, 360, 1)
        output = output.squeeze(-1).permute(0, 2, 1).unsqueeze(1) # (bs, 1, 360, num_filters)
        if self.resize_f > 1:
            output = F.interpolate(output, scale_factor=(1, self.resize_f), mode='bilinear')
        output = self.classification_model(output)
        output = self.attention(output)
        if self.is_tranformer:
            output = output.mean(dim=1)
        else:
            output = self.global_pool(output)
        output = self.head(output)
        return output

    def freeze_filter(self):
        freeze_module(self.filter)

    def unfreeze_filter(self):
        for i, param in enumerate(self.filter.parameters()):
            param.requires_grad = True
        

class MatchedFilterModel1d(nn.Module):

    def __init__(self,
            wave_bank_path,
            filter_height=90,
            resize_factor=1,
            num_filters=1024,
            filter_method='sum',
            in_chans=2,
            num_classes=1,
            hidden_dims=[1024, 512, 256, 128],
            stride=1):

        super().__init__()
        
        wave_bank = torch.load(wave_bank_path, 'cpu')
        wave_bank = wave_bank[:, 180-filter_height//2:180+filter_height//2+1, :]
        num_filters_0, filter_height, filter_width = wave_bank.shape
        if filter_method == 'sum':
            wave_bank = wave_bank.reshape(num_filters, num_filters_0//num_filters, filter_height, filter_width)
            wave_bank = wave_bank.sum(1)
        else:
            wave_bank = wave_bank[:num_filters]
        wave_bank = torch.stack([wave_bank, wave_bank], dim=1) # (num_filters, 2, filter_height, filter_width)
        self.filter = torch.nn.Conv2d(
            in_chans, num_filters, kernel_size=(filter_height, filter_width), stride=1, padding=(filter_height//2, 0), bias=False)
        self.filter.weight = nn.Parameter(wave_bank)
        self.freeze_filter()
        self.resize_f = resize_factor
        del wave_bank
        
        self.cnn = CNN1d(num_filters, num_classes, hidden_dims=hidden_dims, kernel_size=3, stride=stride, reinit=True)

    def forward(self, x):
        output = self.filter(x) # (bs, num_filters, 360, 1)
        output = output.squeeze(-1) # (bs, num_filters, 360)
        output = self.cnn(output)
        return output

    def freeze_filter(self):
        freeze_module(self.filter)

    def unfreeze_filter(self):
        for i, param in enumerate(self.filter.parameters()):
            param.requires_grad = True
        