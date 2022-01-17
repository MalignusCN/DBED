from base.base_model import BaseModel as BaseModel
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import model.resnet as resnet
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.helpers import initialize_weights
from itertools import chain
from einops import rearrange

''' 
-> ResNet BackBone
'''


def aspp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channles),
        nn.ReLU(inplace=True))


class MASPP(nn.Module):
    def __init__(self, in_channels):
        super(MASPP, self).__init__()


        dilations = [1, 3, 5]

        self.aspp1 = aspp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = aspp_branch(in_channels, 256, 3, dilation=dilations[0])
        self.aspp3 = aspp_branch(in_channels, 256, 3, dilation=dilations[0])
        self.aspp4 = aspp_branch(in_channels, 256, 3, dilation=dilations[0])
        self.aspp2_2 = aspp_branch(256, 256, 3, dilation=dilations[1])
        self.aspp3_2 = aspp_branch(256, 256, 3, dilation=dilations[1])
        self.aspp3_3 = aspp_branch(256, 256, 3, dilation=dilations[2])
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x2 = self.aspp2_2(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_2(x3)
        x3 = self.aspp3_3(x3)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, pretrained=True):
        super(ResNet, self).__init__()

        model = resnet.resnet50(pretrained=pretrained, num_classes=1000, dilated=True, multi_grid=True,
                                deep_base=False, norm_layer=nn.BatchNorm2d, in_channels=in_channels, output_stride=output_stride)
        if in_channels == 3:
            self.into_3_channels = nn.Identity()
        else:
            self.into_3_channels = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False)
        initialize_weights(self.into_3_channels)
        # self.CAM = CAM_Module()
        if not pretrained:
            self.layer0 = nn.Sequential(
                # nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                # nn.BatchNorm2d(64),
                # nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                # (224 + 2 * 1 - 3) / 2 + 1= 112
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                # (112 + 2 * 1 - 3) / 1 + 1 = 112
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                # (112 + 2 * 1 - 3) / 1 + 1 = 112
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=1, bias=False),
                # (112 + 2 * 1 - 3) / 1 + 1 = 112
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                # (112 - 3 + 2) / 2 + 1 = 57

            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4



        # if output_stride == 8:
        #     for n, m in self.layer3.named_modules():
        #         if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
        #             m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
        #         elif 'conv2' in n:
        #             m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
        #         elif 'downsample.0' in n:
        #             m.stride = (s3, s3)
        #
        # for n, m in self.layer4.named_modules():
        #     if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
        #         m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
        #     elif 'conv2' in n:
        #         m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
        #     elif 'downsample.0' in n:
        #         m.stride = (s4, s4)

    def forward(self, x):
        # x = PCA_svd(x, 3)
        x = self.into_3_channels(x)
        # x : [batch_size, 4, 224, 224]
        # x = self.CAM(x)
        x = self.layer0(x)
        # x : [batch_size, 64, 57, 57]
        x = self.layer1(x)
        # x : [batch_size, 256, 57, 57]
        low_level_features = x
        x = self.layer2(x)
        mid_level_features = x
        # x : [batch_size, 512, 29, 29]
        x = self.layer3(x)
        temp_high_features = x
        # x : [batch_size, 1024, 29, 29]
        x = self.layer4(x)
        # x : [batch_size, 2048, 29, 29]

        return x, low_level_features, mid_level_features, temp_high_features


class Decoder(nn.Module):
    def __init__(self, low_level_channels, mid_level_features, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_level_features, 128, 1, bias=False)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(1024, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features, mid_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        mid_level_features = self.conv2(mid_level_features)
        mid_level_features = self.relu(self.bn1(mid_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)
        mid_level_features = F.interpolate(mid_level_features, size=(H, W), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, mid_level_features, x), dim=1))
        return x


class CFAMmodule(nn.Module):
    def __init__(self, input_features, mid_features, num_classes):
        super(CFAMmodule, self).__init__()
        self.conv1 = nn.Conv2d(input_features, mid_features, 1, bias=False)
        self.conv2 = nn.Conv2d(input_features, num_classes, 1, bias=False)
        self.conv3 = nn.Conv2d(mid_features, input_features, 1, bias=False)
        initialize_weights(self)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = x2.softmax(dim=1)
        _, C, H, W = x1.size()
        _, N, H, W = x2.size()
        x1 = rearrange(x1, "B C H W -> B C (H W)", C=C, H=H, W=W)
        x2 = rearrange(x2, "B N H W -> B (H W) N", N=N, H=H, W=W)
        x3 = torch.matmul(x1, x2)
        x3 = x3.softmax(dim=-1)
        x2 = rearrange(x2, "B L N -> B N L", N=N)
        x4 = torch.matmul(x3, x2)
        x4 = rearrange(x4, "B C (H W) -> B C H W", H=H, W=W)
        x4 = self.conv3(x4)
        x = x + x4
        return x


class CFAMNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet', pretrained=True,
                 freeze_bn=False, **_):

        super(CFAMNet, self).__init__()
        assert ('resnet' in backbone)

        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=16, pretrained=pretrained)
            low_level_channels = 256
            mid_level_channels = 512
        self.CFAM_conv1 = nn.Conv2d(1024, 256, 1, bias=False)
        self.CFAM_conv2 = nn.Conv2d(2048, 256, 1, bias=False)
        self.CFAM_1 = CFAMmodule(256, 128, 6)
        self.CFAM_2 = CFAMmodule(256, 128, 6)
        self.ASPP = MASPP(in_channels=2048)
        self.decoder = Decoder(low_level_channels, mid_level_channels, num_classes)
        initialize_weights(self.CFAM_conv1)
        initialize_weights(self.CFAM_conv2)
        if freeze_bn: self.freeze_bn()
        # if freeze_backbone:
        #     set_trainable([self.backbone], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        x, low_level_features, mid_level_features, temp_high_features = self.backbone(x)
        temp_high_features = self.CFAM_conv1(temp_high_features)
        temp_high_features = self.CFAM_1(temp_high_features)
        x1 = self.CFAM_conv2(x)
        x1 = self.CFAM_2(x1)
        x2 = self.ASPP(x)
        x = torch.cat((temp_high_features, x1, x2), dim=1)
        x = self.decoder(x, low_level_features, mid_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

    # Two functions to yield the parameters of the backbone
    # & Decoder / ASPP to use differentiable learning rates
    # FIXME: in xception, we use the parameters from xception and not aligned xception
    # better to have higher lr for this backbone

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASPP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()