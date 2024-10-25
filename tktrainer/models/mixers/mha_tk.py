# Copyright (c) 2024, Simran Arora.

from einops import rearrange
import torch
import torch.nn as nn

import thunderkittens as tk

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None


class AttentionFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, k, v):        
        assert q.shape[3] in [64, 128], "TK train currently supports head dim 64 only"

        # SA: Be careful, .contiguous() can remove requires_grad
        q = q.to(torch.bfloat16).contiguous().requires_grad_(True)
        k = k.to(torch.bfloat16).contiguous().requires_grad_(True)
        v = v.to(torch.bfloat16).contiguous().requires_grad_(True)

        o, l_vec = tk.mha_forward(q, k, v, True) 

        ctx.save_for_backward(q, k, v, o, l_vec)
        return o

    @staticmethod
    def backward(ctx, grad_o):        
        assert grad_o.shape[3] in [64, 128], "TK train currently supports head dim 64 only"
        q, k, v, o, l_vec = ctx.saved_tensors

        l_vec = l_vec.contiguous()
        grad_o = grad_o.contiguous()
        q = q.to(torch.bfloat16).contiguous()
        k = k.to(torch.bfloat16).contiguous()
        v = v.to(torch.bfloat16).contiguous()
        o = o.to(torch.bfloat16).contiguous()

        grad_q, grad_k, grad_v = tk.mha_backward(q, k, v, o, l_vec, grad_o, True)

        return grad_q, grad_k, grad_v, None, None 


class MHA_TK(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        qkv_proj_bias=True,
        out_proj_bias=True,
        causal=False,
        layer_idx=None,
        return_residual=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.return_residual = return_residual
  
        self.num_heads = num_heads
        self.num_heads_kv =  num_heads
        self.head_dim = self.embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def forward(
        self,
        x,
        x_kv=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        batch, seqlen = x.shape[:2]
        q = self.q_proj(x).reshape(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)            
        o = AttentionFunction.apply(q, k, v)
        context = o.transpose(1, 2).contiguous()
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)

