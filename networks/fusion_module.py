import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable






# imu: 512, vis: 512 total: 1024
class FuseModule_unvio(nn.Module):
    def __init__(self, channels=1024, reduction=16):
        super(FuseModule_unvio, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, v,i):
        
        B, t, _ = i.shape
        imu_input = i.view(B,t,-1)
        visual_input = v.view(B,t,-1)
        input = torch.cat((visual_input, imu_input), dim=2)
        module_input = input
        #decoding function
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

# imu : 256, vis : 512 total: 768
class Fusion_module_v_sel_vio(nn.Module):
    def __init__(self, opt):
        super(Fusion_module_v_sel_vio, self).__init__()
        self.fuse_method = opt.fuse_method
        self.f_len = opt.i_f_len + opt.v_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len))

    def forward(self, v, i):
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]

# imu : 256, vis : 512 total: 768
class FusionModule_nasvio(nn.Module):
    def __init__(self, temp=None):
        super(FusionModule_nasvio, self).__init__()
        self.fuse_method = 'cat' # soft
        self.f_len = 256 + 512
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len))
            if temp is None:
                self.temp = 1
            else:
                self.temp = temp

    def forward(self, v, i):
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]


if __name__ == "__main__":
    imu_feature = torch.rand(50, 32, 512)
    visual_feature = torch.rand(50, 32, 512)

    model = FuseModule_unvio()
    output = model(visual_feature, imu_feature)
    
    print((output.shape))

