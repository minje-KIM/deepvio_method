import torch 
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from collections import OrderedDict


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2

def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    elif act_func == "gelu":
        return nn.GELU()
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)

class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        use_norm=True,
        norm_func='LN',
        norm_chan_per_group=None,
        use_act=True,
        act_func="relu",
        dropout_rate=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm
        self.norm_func = norm_func
        self.use_act = use_act
        self.act_func = act_func
        self.dropout_rate = dropout_rate

        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        """ modules """
        modules = {}
        # norm layer
        if self.use_norm:
            if norm_func == "BN":
                modules["norm"] = nn.BatchNorm2d(out_channels)
            if norm_func == "LN":
                modules["norm"] = nn.GroupNorm(1, out_channels)
            if norm_func == "GN":
                modules["norm"] = nn.GroupNorm(out_channels//norm_chan_per_group, out_channels)
        else:
            modules["norm"] = None
        # activation
        if use_act:
            modules["act"] = build_activation(
                self.act_func, self.use_norm
            )
        else:
            modules["act"] = None
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # weight
        modules["weight"] = self.weight_op()

        # add modules
        for op in ["weight", "norm", "act"]:
            if modules[op] is None:
                continue
            elif op == "weight":
                # dropout before weight operation
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict(
            {
                "conv": nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=padding,
                    dilation=self.dilation,
                    bias=self.bias,
                )
            }
        )
        return weight_dict

    def forward(self, x):
        # similar to nn.Sequential
        for module in self._modules.values():
            x = module(x)
        return x

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict(
            {
                "conv": nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=padding,
                    dilation=self.dilation,
                    bias=self.bias,
                )
            }
        )
        return weight_dict

    def forward(self, x):
        # similar to nn.Sequential
        for module in self._modules.values():
            x = module(x)
        return x


class VisualEncoderLatency(nn.Module):
    def __init__(self, par):
        super(VisualEncoderLatency, self).__init__()
        # CNN
        self.par = par
        self.batchNorm = par.batch_norm
        # searched net with low latency
        self.conv1 = ConvLayer(6, 8, kernel_size=5, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv2 = ConvLayer(8, 16, kernel_size=3, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv3 = ConvLayer(16, 64, kernel_size=5, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv4 = ConvLayer(64, 64, kernel_size=3, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv4_1 = ConvLayer(64, 64, kernel_size=3, stride=1, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv5 = ConvLayer(64, 192, kernel_size=3, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv5_1 = ConvLayer(192, 64, kernel_size=3, stride=1, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        self.conv6 = ConvLayer(64, 1024, kernel_size=3, stride=2, use_norm=self.batchNorm, norm_func='BN', act_func='lrelu')
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, par.img_w, par.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), par.visual_f_len)

    def forward(self, x, batch_size, seq_len):
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v_f = self.visual_head(x)  # (batch, seq_len, 256)
        return v_f

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6
    
    
if __name__ == "__main__":
    input = torch.rand(50, 32, 11, 6)

    model = VisualEncoderLatency()
    output = model(input)
    
    print((output.shape))