import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_Attention_block, VAE_Residual_block


class VAE_Encoder(nn.Sequential):
    super.__init__(
        nn.Conv2d(3, 128, kernel_size=3, padding=1),
        
        VAE_Residual_block(128, 128),
        VAE_Residual_block(128, 128),
        
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
        
        VAE_Residual_block(128, 256),
        VAE_Residual_block(256, 256),
        
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
        
        VAE_Residual_block(256, 512),
        VAE_Residual_block(512, 512),
        
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
        
        VAE_Residual_block(512, 512),
        VAE_Residual_block(512, 512),
        VAE_Residual_block(512, 512),
        
        VAE_Attention_block(512),
        
        VAE_Residual_block(512, 512),
        
        nn.GroupNorm(32, 512),
        
        nn.SiLU(),
        
        nn.Conv2d(512, 8, kernel_size=3, padding=1),
        nn.Conv2d(8, 8, kernel_size=1, padding=0),
        
    )
    
    def forward(self, x, noise):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # Pad left, right, top, bottom
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = torch.exp(log_variance)
        std = torch.sqrt(variance)
                
        x = mean + std * noise
        # Multuply by a constant (from origianal SD paper)
        x = x * 0.18215
        
        return x