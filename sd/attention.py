import torch
from torch import nn
from torch.nn import functional as F

import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
    def forward(self, x, causal_mask=False):
        input_shape = x.shape
        batch_size, seq_len, embed_dim = input_shape
        
        intermediate_shape = (batch_size, seq_len, self.n_heads, self.head_dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)
        
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, torch.inf)
            
        weight = weight / math.sqrt(self.head_dim)
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        
        return self.out_proj(output)


class CrossAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, cross_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.k_proj = nn.Linear(cross_dim, embed_dim, bias=in_proj_bias)
        self.v_proj = nn.Linear(cross_dim, embed_dim, bias=in_proj_bias)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
    def forward(self, x, y):
        input_shape = x.shape
        batch_size, seq_len, embed_dim = input_shape
        
        intermidiate_shape = (batch_size, -1, self.n_heads, self.head_dim)
        
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        
        q = q.view(intermidiate_shape).transpose(1, 2)
        k = k.view(intermidiate_shape).transpose(1, 2)
        v = v.view(intermidiate_shape).transpose(1, 2)
        
        weight = q @ k.transpose(-1, -2)
        weight = weight / math.sqrt(self.head_dim)
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        
        return output
        