import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import segmentation_models_pytorch as smp
from modules import *

from kuma_utils.torch.modules import AdaptiveConcatPool2d, GeM, AdaptiveGeM
from kuma_utils.torch.utils import freeze_module


'''
Codes from previous G2Net
'''


'''
New
'''
class SimpleCNN(nn.Module):

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

        if custom_preprocess == 'chris_debias':
            self.preprocess = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=(3,31), stride=(1,2), padding=(3//2,31//2)),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1,2), padding=(5//2,5//2)),
                nn.GELU(),
            )
            cls_in_chans = 128
        elif custom_preprocess == 'debias_large':
            self.preprocess = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=(15,31), stride=(1,2), padding=(15//2,31//2)),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1,2), padding=(5//2,5//2)),
                nn.GELU(),
            )
            cls_in_chans = 128
        else:
            self.preprocess = nn.Identity()
            cls_in_chans = in_chans
        
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

        if custom_preprocess == 'chris_debias':
            self.preprocess = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=(3,31), stride=(1,2), padding=(3//2,31//2)),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1,2), padding=(5//2,5//2)),
                nn.GELU(),
            )
            seg_in_chans = 128
        else:
            self.preprocess = nn.Identity()
            seg_in_chans = in_chans

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
                mask =  F.interpolate(mask, size=(mask.shape[2], mask.shape[3]*4))
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
