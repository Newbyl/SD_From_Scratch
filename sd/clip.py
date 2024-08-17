import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layer_norm = nn.LayerNorm(768)
        
    def forward(self, tokens):
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        
        return self.layer_norm(state)


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_tokens):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, embed_dim))
        
    
    def forward(self, tokens):
        return self.token_embedding(tokens) + self.positional_embedding


class CLIPLayer(nn.Module):
    def __init__(self, n_heads, embed_dim):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(n_heads, embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        
        self.linear_1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear_2 = nn.Linear(embed_dim * 4, embed_dim)
        
    def forward(self, x):
        res = x
        
        # Self-attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x = x + res
        
        # FFN
        res = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(x * 1.702) # quickgelu activation
        x = self.linear_2(x)
        
        return x + res