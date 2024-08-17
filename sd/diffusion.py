import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.linear_1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear_2 = nn.Linear(embed_dim * 4, embed_dim)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        
        return x


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
            
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        
        return x


class UNetOutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        
        return x


class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        
    def forward(self, feature, time):
        res = feature
        
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(res)
        
        
class UNetAttentionBlock(nn.Module):
    def __init__(self, n_heads, embed_dim, d_context=768):
        super().__init__()
        
        channels = n_heads * embed_dim
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, in_proj_bias=False)
        
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, channels * 4)
        self.linear_geglu_2 = nn.Linear(channels * 4, channels)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        res_long = x
        
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        x = x.view((n, c, h*w))
        x = x.transpose(-1, -2)
        
        # Norm + Self Attention + residual
        res_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x = x + res_short
        
        # Norm + Cross Attention + residual
        res_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x = x + res_short
        
        # Norm + FFN + residual    
        res_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x = x + res_short
        
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        
        return self.conv_output(x) + res_long
    
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(UNetResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),
            SwitchSequential(UNetResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),
            
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNetResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(640, 640), UNet_AttentionBlock(8, 80)),
        
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNetResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)),
            
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(UNetResidualBlock(1280, 1280)),
            SwitchSequential(UNetResidualBlock(1280, 1280)),
        ])
        
        self.bottle_neck = SwitchSequential(
            UNetResidualBlock(1280, 1280),
            UNetAttentionBlock(8, 160),
            UNetResidualBlock(1280, 1280),
        )
        
        self.decoder = nn.ModuleList([
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            
            SwitchSequential(UNetResidualBlock(2560, 1280), UpSample(1280)),
            
            SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)),
            
            SwitchSequential(UNetResidualBlock(1920, 1280), UNetAttentionBlock(8, 160), UpSample(1280)),
            
            SwitchSequential(UNetResidualBlock(1920, 640), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(1920, 640), UNetAttentionBlock(8, 80)),
            
            SwitchSequential(UNetResidualBlock(960, 640), UNetAttentionBlock(8, 80), UpSample(640)),
            
            SwitchSequential(UNetResidualBlock(960, 320), UNetAttentionBlock(8, 40)),
            SwitchSequential(UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)),
            SwitchSequential(UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)),
        ])
        
        
class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNet_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        time = self.time_embedding(time)
        output = self.unet(latent, context, time)
        output = self.final(output)
        
        return output
        