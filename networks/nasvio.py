import torch 
import torch.nn as nn

class nasvio(nn.Module):
    def __init__(self, par):
        super(nasvio,self).__init__()

        # Initialize networks        
        if par.target == 'flops':
            self.visual_encoder = VisualEncoderFlops(par)
        else:
            self.visual_encoder = VisualEncoderLatency(par)
        
        self.inertial_encoder = InertialEncdoer(par)
        self.pose_regressor = PoseRegressor(par)

        self.par = par
        
    def forward(self, t_x, t_i, prev=None):
        
        
        

class VisualEncoder(nn.Module):
    pass

class VisualEncoderFlops(nn.Module):
    pass

class VisualEncoderLatency(nn.Module):
    pass

class InertialEncdoer(nn.Module):
    pass

class FusionModule(nn.Module):
    pass

class PoseRegressor(nn.Module):
    pass





