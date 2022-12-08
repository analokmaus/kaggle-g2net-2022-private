import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

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
                timm_params.update({
                    'in_chans': 128,
                    'global_pool': '',
                    'num_classes': 0
                })
                self.cnn = timm.create_model(model_name, 
                    pretrained=pretrained, **timm_params)
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
            self.cnn[3][1].update(lam, idx)

        if self.return_spec and lam is not None:
            return self.cnn(spec), spec, lam
        elif self.return_spec:
            return self.cnn(spec), spec
        elif lam is not None:
            return self.cnn(spec), lam
        else:
            return self.cnn(spec)
    