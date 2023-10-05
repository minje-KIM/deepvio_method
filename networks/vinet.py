'''
VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem
https://arxiv.org/abs/1701.08376
'''

import torch
import torch.nn as nn

from .flownet import FlowNetC

class Vinet(nn.Module):
    def __init__(self):
        super(Vinet, self).__init__()

        # 어떻게 파라미터를 정의해야 할까?
        self.PoseRegressor = nn.LSTM(
            input_size = 49165,
            hidden_size = 1,
            num_layers = 1,
            batch_first = True
        )
        
        self.IMUEncoder = nn.LSTM()
        
        self.VisualEncoder = FlowNetC()


    def forward(self):        
        
        