import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_Residual_block(nn.module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        
    def forward(self, x):
        res = x
        
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        
        return x + self.residual_layer(res)
    

class VAE_Attention_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.attention = SelfAttention(in_channels)
        
    def forward(self, x):
        res = x

        m, c, h, w = x.shape
        x = x.view(m, c, h * w)
        x = x.transpose(-1, -2)
        
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((m, c, h, w))
        
        return x + res
        