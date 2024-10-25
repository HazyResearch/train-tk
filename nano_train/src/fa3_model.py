import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from flash_attn_interface import _flash_attn_forward, _flash_attn_backward, flash_attn_func
import wandb
        

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        deterministic=False,
        descale_q=None,
        descale_k=None,
        descale_v=None,
        gqa_parallel=False,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        out, q, k, v, out_padded, softmax_lse, S_dmask = _flash_attn_forward(
            q,
            k,
            v,
            softmax_scale,
            causal,
            window_size,
            descale_q=descale_q,    # None,
            descale_k=descale_k,    # None,
            descale_v=descale_v,    # None,
            gqa_parallel=gqa_parallel,  # False,
        )
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.deterministic = deterministic
        ctx.gqa_parallel = gqa_parallel

        # # HUGE FLAG! OVERRIDING!
        # o = F.scaled_dot_product_attention(
        #     q.permute(0, 2, 1, 3),
        #     k.permute(0, 2, 1, 3),
        #     v.permute(0, 2, 1, 3),
        #     attn_mask=None,
        #     dropout_p=0.0,
        #     is_causal=True
        # ).transpose(1, 2)

        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.deterministic,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None


class CausalSelfFA3Attention(nn.Module):

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
        self.is_causal = config.causal
        print(f"Using Reference CausalSelfFA3Attention -- Causal = {self.is_causal}")

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)

        # Inputs are B, N, H, D
        y, _ = flash_attn_func(q, k, v, causal=self.is_causal)  
        # y = FlashAttnFunc.apply(q, k, v, None, self.is_causal, (-1, -1))

        y = y.contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

