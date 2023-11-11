# encoding: utf-8

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Head import Head
from src.encoder import resnet50, resnet101


class MFFMNet(nn.Module):
    def __init__(self, out_planes, norm_layer, pretrained_model=None):
        super(MFFMNet, self).__init__()
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer, deep_stem=True, stem_width=64,
                                 embed_dims=[256, 512, 1024, 2048])

        self.dilate = 2  # 使得最后一层变成 (30,40)

        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.head = Head(out_planes, norm_layer, 0.1)

    def forward(self, data, depth):
        b, c, h, w = data.shape
        blocks, merges = self.backbone(data, depth)

        pred, aux_fm = self.head(merges)

        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        aux_fm = F.interpolate(aux_fm, size=(h, w), mode='bilinear', align_corners=True)

        return pred, aux_fm

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


if __name__ == '__main__':
    device = "cpu"
    model_ = MFFMNet(37,
                     pretrained_model=None,
                     norm_layer=nn.BatchNorm2d)  # .to(device)
    # print(model_)
    # in_batch, inchannel_rgb, in_h, in_w = 2, 3, 240, 320
    # inchannel_depth = 1
    # rgb = torch.randn(in_batch, inchannel_rgb, in_h, in_w,dtype=torch.float32).to(device)
    # depth = torch.randn(in_batch, inchannel_depth, in_h, in_w,dtype=torch.float32).to(device)
    '''
    resnet50、resnet101
    
    layer1 torch.Size([2, 256, 120, 160]) torch.Size([2, 256, 120, 160])    32
    layer2 torch.Size([2, 512, 60, 80]) torch.Size([2, 512, 60, 80])        16
    layer3 torch.Size([2, 1024, 30, 40]) torch.Size([2, 1024, 30, 40])      8
    layer4 torch.Size([2, 2048, 30, 40]) torch.Size([2, 2048, 30, 40])      8
    '''
    # out = model_(rgb, depth)
    # print(out[0].shape)
    # print(out[0].dtype)# torch.float32
    from utils.utils import compute_speed

    image_w = 640
    image_h = 480
    batch_size = 2


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    compute_speed(model_, (batch_size, 3, image_h, image_w), (batch_size, 1, image_h, image_w), device, 1)

    total = sum([param.nelement() for param in model_.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    '''
    
    Elapsed Time: [1.76 s / 10 iter]
    Speed Time: 175.97 ms / iter   FPS: 5.68
    '''
