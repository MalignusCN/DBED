import torch
from torch import nn


class CAM_DANet(nn.Module):
    """ Channel attention module from DANet"""
    def __init__(self):
        super(CAM_DANet, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        # B, C, H*W

        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # B, H*W, C
        energy = torch.bmm(proj_query, proj_key)
        # B, C, C
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy

        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_CBAM_Temp(nn.Module):
    def __init__(self, in_planes=4):
        super(CAM_CBAM_Temp, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes * 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes * 2, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class CAM_CBAM(nn.Module):
    def __init__(self, in_planes=4):
        super(CAM_CBAM, self).__init__()
        self.ca = CAM_CBAM_Temp(in_planes)

    def forward(self, x):
        out = self.ca(x) * x
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_planes=4):
        super(ChannelAttentionModule, self).__init__()
        self.ca_1 = CAM_DANet()
        self.ca_2 = CAM_CBAM(in_planes)

    def forward(self, x):
        ca_1 = self.ca_1(x)
        ca_2 = self.ca_2(x)
        feature = ca_1 + ca_2

        return feature
