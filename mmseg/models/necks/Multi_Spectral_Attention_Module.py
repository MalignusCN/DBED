# Copyright (c) OpenMMLab. All rights reserved.
from turtle import forward
from cv2 import TERM_CRITERIA_EPS
from numpy import isin
import torch.nn as nn
import torch
from mmcv.cnn import build_norm_layer
import math
import torch
import torch.nn as nn
from ..builder import NECKS

from mmcv.cnn import ConvModule, xavier_init

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        c2wh = dict([(256, 56), (512, 28), (1024, 14), (2048, 7)])
        self.reduction = reduction
        self.dct_h = c2wh[channel]
        self.dct_w = c2wh[channel]

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (self.dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (self.dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(self.dct_h, self.dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter


class CAM_CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CAM_CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SAM_CBAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM_CBAM, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                xavier_init(m, distribution='uniform')

class CAM_DANet(nn.Module):
    
    def __init__(self):
        super(CAM_DANet, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, x):
        '''
            inputs:
               x: input feature maps (B, C, H, W)
            returns :
               out: attention value + input feature (B, C, H, W)
        '''

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


class eca_layer(nn.Module):
    
    def __init__(self):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)

@NECKS.register_module()
class MSAM(nn.Module):
    """
    Multi-spectral attention module

    Args:
        plan(int): 
            Which decide the methods we adopt in MSAM
            0: (x_nir + x_rgb) -> ECANet
            1: concat (x_nir, x_rgb) -> DANet -> nn.Conv2D (kernel=1) adjust the dim
            2: x_nir + x_rgb -> nn.Conv2D(kernel=3)
            3. concat (x_nir, x_rgb) -> nn.Conv2D (kernel=3)
            4: one by one concat (x_nir, x_rgb) -> MSAM -> nn.Conv2D(kernel=1) adjust the dim
            5: (x_nir + x_rgb) -> DANet
            6: (x_nir + x_rgb) -> CBAM
            7: x_nir + x_rgb
            8: x_nir + x_rgb -> FcaNet
            9: (x_nir + x_rgb) -> ECANet + DANet(channel attention) (MSAM)
    """

    def __init__(self, plan=0, input_feature_dims=[]):
        super(MSAM, self).__init__()
        self.plan = plan
        if self.plan == 0:
            self.ca = eca_layer()
        if len(input_feature_dims) > 0:
            if self.plan == 1 or self.plan == 4:
                self.convlist = nn.ModuleList(
                    [nn.Conv2d(input_feature_dim * 2, input_feature_dim, kernel_size=1) \
                        for input_feature_dim in input_feature_dims]
                )
            if self.plan == 2:
                self.convlist = nn.ModuleList(
                    [nn.Conv2d(input_feature_dim, input_feature_dim, kernel_size=3, padding=1) \
                        for input_feature_dim in input_feature_dims]
                )
            if self.plan == 3:
                self.convlist = nn.ModuleList(
                    [nn.Conv2d(input_feature_dim * 2, input_feature_dim, kernel_size=3, padding=1) \
                        for input_feature_dim in input_feature_dims]
                )
            if self.plan == 5:
                self.ca = CAM_DANet()
                self.pa = nn.ModuleList(
                    [PAM_Module(input_feature_dim) for input_feature_dim in input_feature_dims]
                )
            if self.plan == 6:
                self.ca = nn.ModuleList(
                    [CAM_CBAM(input_feature_dim) for input_feature_dim in input_feature_dims]
                )
                self.sa = nn.ModuleList(
                    [SAM_CBAM() for _ in input_feature_dims]
                )
            if self.plan == 8:
                self.ca = nn.ModuleList(
                    [MultiSpectralAttentionLayer(input_feature_dim) for input_feature_dim in input_feature_dims]
                )
            if self.plan == 9:
                self.ca1 = eca_layer()
                self.ca2 = CAM_DANet()
            if self.plan == 1 or self.plan == 4:
                self.ca1 = eca_layer()
                self.ca2 = CAM_DANet()
                self.convlist = nn.ModuleList(
                    [nn.Conv2d(input_feature_dim * 2, input_feature_dim, kernel_size=1) \
                        for input_feature_dim in input_feature_dims]
                )

    def forward(self, x_nir, x_rgb):
        new_tuple = ()
        if isinstance(x_nir, tuple) or isinstance(x_nir, list):
            if self.plan == 0:
                for idx in range(len(x_nir)):
                    temp_x = x_nir[idx] + x_rgb[idx]
                    feature = self.ca(temp_x)
                    new_tuple = new_tuple + (feature, ) 
            if self.plan == 1:
                for idx in range(len(x_nir)):
                    temp_x = torch.cat((x_nir[idx], x_rgb[idx]), 1)
                    feature_1 = self.ca1(temp_x)
                    feature_2 = self.ca2(temp_x)
                    feature = feature_1 + feature_2
                    feature = self.convlist[idx](feature)
                    new_tuple = new_tuple + (feature, )
            if self.plan == 2:
                for idx in range(len(x_nir)):
                    temp_x = x_nir[idx] + x_rgb[idx]
                    feature = self.convlist[idx](temp_x)
                    new_tuple = new_tuple + (feature, )
            if self.plan == 3:
                for idx in range(len(x_nir)):
                    temp_x = torch.cat((x_nir[idx], x_rgb[idx]), 1)
                    feature = self.convlist[idx](temp_x)
                    new_tuple = new_tuple + (feature, )
            if self.plan == 4:
                for idx in range(len(x_nir)):
                    temp_x_temp = torch.cat((x_nir[idx], x_rgb[idx]), 1)
                    temp_x = temp_x_temp.view(temp_x_temp.size(0), 2, temp_x_temp.size(1)//2, temp_x_temp.size(2), temp_x_temp.size(3)).permute(\
                        0, 2, 1, 3, 4).contiguous().view_as(temp_x_temp)
                    feature_1 = self.ca1(temp_x)
                    feature_2 = self.ca2(temp_x)
                    feature = feature_1 + feature_2
                    feature = self.convlist[idx](feature)
                    new_tuple = new_tuple + (feature, )
            if self.plan == 5:
                for idx in range(len(x_nir)):
                    temp_x = x_nir[idx] + x_rgb[idx]
                    feature_ca = self.ca(temp_x)
                    feature_pa = self.pa[idx](temp_x)
                    feature = feature_ca + feature_pa
                    new_tuple = new_tuple + (feature, ) 
            if self.plan == 6:
                for idx in range(len(x_nir)):
                    temp_x = x_nir[idx] + x_rgb[idx]
                    feature = self.ca[idx](temp_x) * temp_x
                    feature = self.sa[idx](feature) * feature
                    new_tuple = new_tuple + (feature, ) 
            if self.plan == 7:
                for idx in range(len(x_nir)):
                    temp_x = x_nir[idx] + x_rgb[idx]
                    # feature = self.ca[idx](temp_x)
                    new_tuple = new_tuple + (temp_x, ) 
            if self.plan == 8:
                for idx in range(len(x_nir)):
                    temp_x = x_nir[idx] + x_rgb[idx]
                    feature = self.ca[idx](temp_x)
                    new_tuple = new_tuple + (feature, ) 
            if self.plan == 9:
                for idx in range(len(x_nir)):
                    temp_x = x_nir[idx] + x_rgb[idx]
                    feature_1 = self.ca1(temp_x)
                    feature_2 = self.ca2(temp_x)
                    feature = feature_1 + feature_2
                    new_tuple = new_tuple + (feature, ) 
        # if isinstance(x_nir, )
        return new_tuple

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.ModuleList):
                for temp_m in m:
                    if isinstance(temp_m, nn.Conv2d) or isinstance(temp_m, nn.Conv1d):
                        xavier_init(temp_m, distribution='uniform')