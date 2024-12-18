import logging
import math
import sys

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.ao.nn.quantized import FloatFunctional

from nets.MobileNetV2 import MobileNetV2, InvertedResidual


class MobileNetV2_unet(nn.Module):
    def __init__(self, pre_trained='weights/mobilenet_v2.pth.tar'):
        super(MobileNetV2_unet, self).__init__()

        # Quantized floating point operations
        self.skip_add = FloatFunctional()
        self.skip_cat = FloatFunctional()

        # Main backbone
        self.backbone = MobileNetV2()

        # Replace transpose convolutions with regular convs + interpolate
        self.dconv1 = nn.Sequential(
            nn.Conv2d(1280, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU6(inplace=True)
        )

        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.Sequential(
            nn.Conv2d(96, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.Sequential(
            nn.Conv2d(32, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True)
        )

        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.Sequential(
            nn.Conv2d(24, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.conv_last = nn.Sequential(
            nn.Conv2d(16, 3, 1),
            nn.BatchNorm2d(3),
            nn.ReLU6(inplace=True)
        )

        self._init_weights()

        if pre_trained is not None:
            self.backbone.load_state_dict(torch.load(pre_trained))

    def forward(self, x):
        # Encoder path
        for n in range(0, 2):
            x = self.backbone.features[n](x)
        x1 = x
        
        for n in range(2, 4):
            x = self.backbone.features[n](x)
        x2 = x
        
        for n in range(4, 7):
            x = self.backbone.features[n](x)
        x3 = x
        
        for n in range(7, 14):
            x = self.backbone.features[n](x)
        x4 = x
        
        for n in range(14, 19):
            x = self.backbone.features[n](x)
        x5 = x
        
        # Decoder path with explicit size matching
        x = self.dconv1(x)
        x = interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=False)
        up1 = torch.cat([x4, x], dim=1)
        up1 = self.invres1(up1)
        
        x = self.dconv2(up1)
        x = interpolate(x, size=x3.shape[2:], mode='bilinear', align_corners=False)
        up2 = torch.cat([x3, x], dim=1)
        up2 = self.invres2(up2)
        
        x = self.dconv3(up2)
        x = interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        up3 = torch.cat([x2, x], dim=1)
        up3 = self.invres3(up3)
        
        x = self.dconv4(up3)
        x = interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        up4 = torch.cat([x1, x], dim=1)
        up4 = self.invres4(up4)
        
        x = self.conv_last(up4)
        x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    # Debug
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = MobileNetV2_unet(pre_trained=None)
    net(torch.randn(1, 3, 224, 224))
