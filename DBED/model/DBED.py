import base.base_model
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import model.resnet as resnet
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.helpers import initialize_weights
from itertools import chain
from model.swin_transformer_micro import *
from model.Channel_attention import ChannelAttentionModule

''' 
-> ResNet BackBone
'''


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet50', pretrained=True):
        super(ResNet, self).__init__()
        if backbone == "resnet50":
            model = resnet.resnet50(pretrained=pretrained, num_classes=1000, dilated=True, multi_grid=True,
                                    deep_base=False, norm_layer=nn.BatchNorm2d, in_channels=in_channels,
                                    output_stride=output_stride)
        elif backbone == "resnet101":
            model = resnet.resnet101(pretrained=pretrained, num_classes=1000, dilated=True, multi_grid=True,
                                     deep_base=False, norm_layer=nn.BatchNorm2d, in_channels=in_channels,
                                     output_stride=output_stride)
        elif backbone == "resnet152":
            model = resnet.resnet152(pretrained=pretrained, num_classes=1000, dilated=True, multi_grid=True,
                                     deep_base=False, norm_layer=nn.BatchNorm2d, in_channels=in_channels,
                                     output_stride=output_stride)
        else:
            print("This backbone didn't exist, use resnet50 instead.")
            model = resnet.resnet50(pretrained=pretrained, num_classes=1000, dilated=True, multi_grid=True,
                                    deep_base=False, norm_layer=nn.BatchNorm2d, in_channels=in_channels,
                                    output_stride=output_stride)
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

        if pretrained:
            self.layer0.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

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
        # x : [batch_size, 512, 29, 29]
        x = self.layer3(x)
        # x : [batch_size, 1024, 29, 29]
        x = self.layer4(x)
        # x : [batch_size, 2048, 29, 29]

        return x, low_level_features


''' 
-> (Aligned) Xception BackBone
Pretrained model from https://github.com/Cadene/pretrained-models.pytorch
by Remi Cadene
'''


class SeparableConv2d(nn.Module):  # depthwise
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, exit_flow=False, use_1st_relu=True):
        super(Block, self).__init__()

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        self.relu = nn.ReLU(inplace=True)

        rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=1, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(out_channels, out_channels, 3, stride=stride, dilation=dilation))
        rep.append(nn.BatchNorm2d(out_channels))

        if exit_flow:
            rep[3:6] = rep[:3]
            rep[:3] = [
                self.relu,
                SeparableConv2d(in_channels, in_channels, 3, 1, dilation),
                nn.BatchNorm2d(in_channels)]

        if not use_1st_relu: rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        output = self.rep(x)
        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        x = output + skip
        return x


''' 
-> The Atrous Spatial Pyramid Pooling
'''


def aspp_branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channles),
        nn.ReLU(inplace=True))


class ASPP(nn.Module):
    def __init__(self, in_channels, output_stride):
        super(ASPP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = aspp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = aspp_branch(in_channels, 256, 3, dilation=dilations[1])
        self.aspp3 = aspp_branch(in_channels, 256, 3, dilation=dilations[2])
        self.aspp4 = aspp_branch(in_channels, 256, 3, dilation=dilations[3])

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
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x


class ASPPTwo(nn.Module):
    def __init__(self, in_channels, output_stride):
        super(ASPPTwo, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = aspp_branch(in_channels, 64, 1, dilation=dilations[0])
        self.aspp2 = aspp_branch(in_channels, 64, 3, dilation=dilations[1])
        self.aspp3 = aspp_branch(in_channels, 64, 3, dilation=dilations[2])
        self.aspp4 = aspp_branch(in_channels, 64, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(64 * 5, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x

''' 
-> Decoder
'''

class DecoderTwo(nn.Module):
    def __init__(self, low_level_channels, low_level_channels_new, num_classes):
        super(DecoderTwo, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(low_level_channels_new, 16, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.CAM = ChannelAttentionModule(64 + 320)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(64 + 320, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features, low_level_features_new):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        # [B, 48, H/4, W/4]
        low_level_features_new = self.conv2(low_level_features_new)
        low_level_features_new = self.relu(self.bn2(low_level_features_new))
        # [B, 16, H/4, W/4]
        low_level_features = torch.cat((low_level_features, low_level_features_new), dim=1)
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = torch.cat((low_level_features, x), dim=1)
        x = self.CAM(x)
        x = self.output(x)
        return x


class DBED(base.base_model.BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='swin_t',
                 freeze_bn=False, pretrained=True, **_):

        super(DBED, self).__init__()
        assert ('swin_b' or 'swin_s' or 'swin_t' or 'swin_l' or "swin_res" in backbone)
        if backbone == "swin_b":
            self.backbone_1 = swin_b(pretrained=True, in_chans=3)
            self.backbone_2 = swin_b(pretrained=False, in_chans=1)
            low_level_channels = 128
            self.ASPP = ASPP(in_channels=1024, output_stride=16)
            self.ASPPTwo = ASPPTwo(in_channels=1024, output_stride=16)
            self.decoder = DecoderTwo(low_level_channels, low_level_channels, num_classes)
        if pretrained:
            self.backbone_1.eval()
        # if freeze_backbone:
        #     set_trainable([self.backbone], False)

    def forward(self, x):
        # X = B, C, H, W
        # x = self.CAM(x)
        x1 = x[:, 1:, :, :]
        # [B, 3, H, W]
        x2 = x[:, 0, :, :].unsqueeze(dim=1)
        # [B, 1, H, W]
        H, W = x.size(2), x.size(3)
        x1, low_level_features = self.backbone_1(x1)
        # X1: [B, 1024, H / 16, W / 16], low_level_features: [B, 128, H/4, W/4]
        x2, low_level_features_new = self.backbone_2(x2)
        # X2: [B, 1024, H / 16, W / 16], low_level_features: [B, 128, H/4, W/4]
        x1 = self.ASPP(x1)
        # x1: [B, 256, H/16, W/16]
        x2 = self.ASPPTwo(x2)
        # X2: [B, 64, H/16, W/16]
        x = torch.cat((x1, x2), dim=1)
        # x: [B, 320, H/16, W/16]
        x = self.decoder(x, low_level_features, low_level_features_new)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

    # Two functions to yield the parameters of the backbone
    # & Decoder / ASPP to use differentiable learning rates
    # FIXME: in xception, we use the parameters from xception and not aligned xception
    # better to have higher lr for this backbone

    def get_backbone_params(self):
        return chain(self.backbone_1.parameters(), self.backbone_2.parameters())

    def get_decoder_params(self):
        return chain(self.ASPP.parameters(), self.ASPPTwo.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

    def get_decoder_modules(self):
        return self.decoder.named_modules()


