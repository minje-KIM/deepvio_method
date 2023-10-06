import torch 
import torch.nn as nn
import numpy as np  
import torch.nn.init as init

from torch.autograd import Variable

from fusion_module import FuseModule_unvio



class PoseNet_unvio(nn.Module):
    '''
    Fuse both features and output the 6 DOF camera pose
    '''
    def __init__(self, input_size=1024):
        super(PoseNet_unvio, self).__init__()

        self.se = FuseModule_unvio(input_size, 16)

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=1024,
                           num_layers=2,
                           batch_first=True)

        self.fc1 = nn.Sequential(nn.Linear(1024, 6))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                init.xavier_normal_(m.all_weights[0][0], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[0][1], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[1][0], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[1][1], gain=np.sqrt(1))
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data, gain=np.sqrt(1))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, visual_fea, imu_fea):
        self.rnn.flatten_parameters()
        
        if imu_fea is not None:
            B, t, _ = imu_fea.shape
            imu_input = imu_fea.view(B, t, -1)
            visual_input = visual_fea.view(B, t, -1)
            inpt = torch.cat((visual_input, imu_input), dim=2)
        else:
            inpt = visual_fea
            
        #inpt = self.se(visual_fea, imu_fea)
        out, (h, c) = self.rnn(inpt)
        out = 0.01 * self.fc1(out)
        return out

class Pose_RNN_v_sel_vio(nn.Module):
    def __init__(self):
        super(Pose_RNN_v_sel_vio, self).__init__()

        # The main RNN network
        f_len = 512 + 256
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size= 1024,
            num_layers=2,
            dropout= 0.2,
            batch_first=True)

        # The output networks
        self.rnn_drop_out = nn.Dropout(0.2)
        self.regressor = nn.Sequential(
            nn.Linear(1024 , 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, fv, fv_alter, fi, dec, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
        # Select be tween fv and fv_alter
        # where the fv_alter comes from???
        v_in = fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:] if fv_alter is not None else fv
        fused = self.fuse(v_in, fi)
        
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc

class PoseRNN_nasvio(nn.Module):
    def __init__(self):
        super(PoseRNN_nasvio, self).__init__()

        # The main RNN network
        f_len = 512 + 256
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=1024,
            num_layers=2,
            dropout=0.2,
            batch_first=True)

        # The output networks
        self.rnn_drop_out = nn.Dropout(0.2)
        # self.regressor = nn.Sequential(
        #    nn.Linear(par.rnn_hidden_size, 6))
        self.regressor = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, fused_f, prev=None):

        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())

        #batch_size = fused_f.shape[0]
        #seq_len = fused_f.shape[1]
        
        #self.rnn.flatten_parameters()
        out, hc = self.rnn(fused_f) if prev is None else self.rnn(fused_f, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)
        angle = pose[:, :, :3]
        trans = pose[:, :, 3:]

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return angle, trans, hc, out
    

if __name__ == "__main__":
    fused_feature = torch.rand(50,32,768)
    
    imu_feature = torch.rand(50, 32, 512)
    visual_feature = torch.rand(50, 32, 512)
    
    model = PoseNet()
    output = model(visual_feature, imu_feature)
    
    print((output.shape))