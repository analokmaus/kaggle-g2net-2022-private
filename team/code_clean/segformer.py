import torch
import torch.nn as nn
import torch.nn.functional as F

from segformer_head import *
from mix_transformer import *

def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)

# Cell
class PixelShuffle_ICNR(nn.Sequential):
    def __init__(self, ni, nf=None, scale=2, blur=True, act=nn.ReLU):
        super().__init__()
        nf = ni if nf is None else nf
        layers = [nn.Conv2d(ni, nf*(scale**2), 1), nn.BatchNorm2d(nf*(scale**2)), 
                  nn.ReLU(inplace=True), nn.PixelShuffle(scale)]
        layers[0].weight.data.copy_(icnr_init(layers[0].weight.data))
        if blur: layers += [nn.ReplicationPad2d((1,0,1,0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)

        
class SegFormer(nn.Module):
    def __init__(self, arch='B0', pre=None, num_classes=1, ps=0.1, use_checkpoint=False, **kwargs):
        super().__init__()
        if arch == 'B0':
            self.backbone = mit_b0(use_checkpoint=use_checkpoint)
            self.decode_head = SegFormerHead(feature_strides=[4, 8, 16, 32],
                            in_channels=[32, 64, 160, 256],embedding_dim=256)
        elif arch == 'B1':
            self.backbone = mit_b1(use_checkpoint=use_checkpoint)
            self.decode_head = SegFormerHead(feature_strides=[4, 8, 16, 32],
                            in_channels=[64, 128, 320, 512],embedding_dim=768)
        elif arch == 'B2':
            self.backbone = mit_b2(use_checkpoint=use_checkpoint)
            self.decode_head = SegFormerHead(feature_strides=[4, 8, 16, 32],
                            in_channels=[64, 128, 320, 512],embedding_dim=768)
        elif arch == 'B3':
            self.backbone = mit_b3(use_checkpoint=use_checkpoint)
            self.decode_head = SegFormerHead(feature_strides=[4, 8, 16, 32],
                            in_channels=[64, 128, 320, 512],embedding_dim=768)
        elif arch == 'B4':
            self.backbone = mit_b4(use_checkpoint=use_checkpoint)
            self.decode_head = SegFormerHead(feature_strides=[4, 8, 16, 32],
                            in_channels=[64, 128, 320, 512],embedding_dim=768)
        elif arch == 'B5':
            self.backbone = mit_b5(use_checkpoint=use_checkpoint)
            self.decode_head = SegFormerHead(feature_strides=[4, 8, 16, 32],
                            in_channels=[64, 128, 320, 512],embedding_dim=768)
        else:
            raise NotImplementedError
        if pre is not None:
            print(f'loading {pre}')
            self.load_state_dict(torch.load(pre)['state_dict'],strict=False)
        
        self.final_conv = nn.Sequential(
            PixelShuffle_ICNR(self.decode_head.embedding_dim,self.decode_head.embedding_dim//4),
            nn.Dropout2d(ps), nn.Conv2d(self.decode_head.embedding_dim//4,num_classes, 3, padding=1))
        self.up_result=2
        
    def forward(self,x):
        x = self.backbone(x)
        x = self.decode_head(x)
        x = F.interpolate(self.final_conv(x),scale_factor=self.up_result,mode='bilinear',align_corners=True)
        return x

