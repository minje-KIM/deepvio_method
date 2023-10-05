import torch 
import torch.nn as nn

class v_selective_vio(nn.Module):
    def __init__(self, opt):
        super(v_selective_vio, self).__init__()

        self.visual_encoder = VisualEncoder(opt)
        self.inertial_encoder = InertialEncoder(opt)
        self.pose_regressor = PoseRegressor(opt)


    def forward(self, img, imu):
        feature
        


class VisualEncoder(nn.Module):
    pass

class InertialEncoder(nn.Module):
    def __init__(self, opt):
        super(InertialEncoder, self).__init__()

        self.inertial_encoder = nn.Sequential(


        )

    def forward(self, x):
        # Input : (batch_size, seq_len, 11, 6)
        # We have 11 IMU measurements between every two consecutive images
        
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))          # x: (batch_size x seq_len, 11, 6)
        x = self.inertial_encoder(x.permute(0,2,1))                     # x: (batch_size x seq_len, 6, 11)
        

class FusionModule(nn.Module):
    pass

class PoseRegressor(nn.Module):
    def __init__(self, opt):
        super(PoseRegressor, self).__init__()
        
        # total feature length = visual feature length + IMU feature length
        f_len = opt.v_f_len + opt.i_f_len
        
        self.rnn = nn.LSTM(input_size=f_len, hidden_size=opt.rnn_hidden_size, 
                            num_layers=2,dropout=opt.rnn_dropout_between, batch_first=True)
        
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))
        
    def forward(self, )