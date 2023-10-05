import torch 
import torch.nn as nn
from torch.autograd import Variable

# input (N,6) / output (N,6)
class InertialEncoder_vinet(nn.Module):
    def __init__(self):
        super(InertialEncoder_vinet, self).__init__()
        
        self.rnnIMU = nn.LSTM(
            input_size=6, 
            hidden_size=6,
            num_layers=2,
            batch_first=True)

    def forward(self, x):
        x = self.rnnIMU(x)
        return x[0]

# input (N,4,6) / output (4,256)
class RecImu_selective_vio(nn.Module):
    
    # x_dim=6, h_dim=128, batch_size=4 , n_layers=2, output_dim=256
    #def __init__(self, x_dim, h_dim, batch_size, n_layers, output_dim):
    def __init__(self):
        super(RecImu_selective_vio, self).__init__()

        self.x_dim = 6
        self.h_dim = 128
        self.n_layers = 2
        self.lstm = nn.LSTM(input_size=self.x_dim, hidden_size=self.h_dim, num_layers=self.n_layers, 
                            dropout=0.2, bidirectional=True)
        self.batch_size = 4
        self.dropout = nn.Dropout(0.2)
        self.hidden = self.init_hidden()
        self.output_dim = 256

    def init_hidden(self):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.n_layers*2, self.batch_size, self.h_dim)).cuda(),
                    Variable(torch.zeros(self.n_layers*2, self.batch_size, self.h_dim)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers*2, self.batch_size, self.h_dim)),
                    Variable(torch.zeros(self.n_layers*2, self.batch_size, self.h_dim)))

    def init_test_hidden(self):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.n_layers*2, 1, self.h_dim)).cuda(),
                    Variable(torch.zeros(self.n_layers*2, 1, self.h_dim)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers*2, 1, self.h_dim)),
                    Variable(torch.zeros(self.n_layers*2, 1, self.h_dim)))

    def forward(self, x):

        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #lstm_out = self.dropout(lstm_out)
        
        # lstm's last output!
        result = lstm_out[-1].view(-1, self.h_dim * 2)

        result = result.view(1, -1, self.output_dim)

        return result

# input (50,32,11,6) / output (50,32,512)
class ImuNet_unvio(nn.Module):
    '''
    Encode imus into imu feature
    '''
    def __init__(self, input_size=6, hidden_size=512):
        super(ImuNet_unvio, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=2,
                           batch_first=True)
        
    def forward(self, imus):
        self.rnn.flatten_parameters()
        x = imus
        B, t, N, _ = x.shape  # B T N 6
        x = x.reshape(B * t, N, -1)  # B T*N 6
        out, (h, c) = self.rnn(x)  # B*T 1000
        out = out[:, -1, :] # take the last element
        return out.reshape(B, t, -1)

# input (50,32,11,6) / output (50,32,256)
class Inertial_encoder_v_sel_vio(nn.Module):
    def __init__(self):
        super(Inertial_encoder_v_sel_vio, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.Dropout(opt.imu_dropout)
            )
        self.proj = nn.Linear(256 * 1 * 11, 256)

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
        x = self.encoder_conv(x.permute(0, 2, 1))                 # x: (N x seq_len, 64, 11)
        out = self.proj(x.view(x.shape[0], -1))                   # out: (N x seq_len, 256)
        return out.view(batch_size, seq_len, 256)

# input (50,32,11,6) / output (50,32,256)
class InertialEncoder_nasvio(nn.Module):
    def __init__(self):
        super(InertialEncoder_nasvio, self).__init__()
        self.method = 'bi-LSTM' # par.imu_method
        
        if self.method == 'bi-LSTM':
            self.rnn_imu_head = nn.Linear(6, 128)
            self.encoder = nn.LSTM(
                input_size=128,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                batch_first=True,
                bidirectional=True)
            
        elif self.method == 'conv':
            self.encoder_conv = nn.Sequential(
                nn.Conv1d(6, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0))
            len_f = 10 + 1 + 0
            #len_f = (len_f - 1) // 2 // 2 + 1
            self.proj = nn.Linear(256 * 1 * len_f, 256)

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0] # N
        seq_len = x.shape[1]    # seq_len
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
        
        if self.method == 'bi-LSTM':
            x = self.rnn_imu_head(x)    # x: (N x seq_len, 11, 128)
            x, hc = self.encoder(x)     # x: (N x seq_len, 11, 2, 128)
            x = x.view(x.shape[0], x.shape[1], 2, -1)
            out = torch.cat((x[:, 0, 0, :], x[:, -1, 1, :]), -1)   # out: (N x seq_len, 256)
            return out.view(batch_size, seq_len, 256)
        
        elif self.method == 'conv':
            x = self.encoder_conv(x.permute(0, 2, 1))    # x: (N x seq_len, 64, 11)
            out = self.proj(x.view(x.shape[0], -1))      # out: (N x seq_len, 256)
            return out.view(batch_size, seq_len, 256)


if __name__ == "__main__":
    input = torch.rand(50, 32, 11, 6)
    #input = input.view(input.size(0), input.size(1), -1)
    #input = torch.rand(145,4, 6)
    #input = input.to('cuda:0')

    
    
    model = RecImu_selective_vio()
    #model = model.to('cuda:0')  

    output = model(input)
    
    print((output.shape))





