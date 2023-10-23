import sys 

# from filepaths import * 
from sequence_models.structure import Attention1d
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import Ridge

#####
# Downstream Model architectures
#####

class ESMAttention1d(nn.Module):
    """Outputs of the ESM model with the attention1d"""
    def __init__(self, max_length, d_embedding, target_class=1): # [batch x sequence(751) x embedding (1280)] --> [batch x embedding] --> [batch x 1]
        super(ESMAttention1d, self).__init__()
        self.attention1d = Attention1d(in_dim=d_embedding) # ???
        self.linear = nn.Linear(d_embedding, d_embedding)
        self.relu = nn.ReLU()
        self.final = nn.Linear(d_embedding, target_class)
        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, x, input_mask):
#        print(x.shape, input_mask.unsqueeze(-1).shape)
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.relu(self.linear(x))
        x = self.final(x)
    #    x = self.softmax(x)
        
        return x
    
    
class ESMAttention1dMean(nn.Module):
    """Attention1d removed, leaving a basic linear model"""
    def __init__(self, d_embedding, target_class=1): # [batch x embedding (1280)]  --> [batch x 1]
        super(ESMAttention1dMean, self).__init__()
        self.linear = nn.Linear(d_embedding, d_embedding)
        self.relu = nn.ReLU()
        self.final = nn.Linear(d_embedding, target_class)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.final(x)
#        x = self.softmax(x)
        return x
    
    
class MaskedConv1d(nn.Conv1d):
    """ A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                           groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)
    
    