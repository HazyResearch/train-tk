import torch.nn as nn
# from flash_attn import flash_attn_qkvpacked_func
from flash_attn_interface import _flash_attn_forward, _flash_attn_backward, flash_attn_func


class CausalSelfFA2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        print(f"Using Reference CausalSelfFA2Attention")

    def forward(self, x):
        B, T, C = x.size() 
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)

        y, _ = flash_attn_func(q, k, v, causal=True) 
        y = y.contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y