import thunderkittens as tk
import torch
import torch.nn as nn
from torch.autograd import Function
import wandb
import os

# from flash_attn_interface import flash_attn_func
# from flash_attn import flash_attn_func

class AttentionFunction(Function):
    def forward(ctx, q, k, v, o, l_vec,  d_vec):        
        assert q.shape[3] == 64, "TK train currently supports head dim 64 only"
        
        l_vec.zero_()
        d_vec.zero_()
        o.zero_()
        q = q.to(torch.bfloat16).contiguous()
        k = k.to(torch.bfloat16).contiguous()
        v = v.to(torch.bfloat16).contiguous()
        o = o.to(torch.bfloat16).contiguous()

        torch.cuda.synchronize()
        tk.mha_forward(q, k, v, o, l_vec, True) 
        torch.cuda.synchronize()

        y_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0, is_causal=True
        ) 
        # o = y_ref # SUPER FLAG

        # ###### AARYAN #####
        # q_ = q.to(torch.float64)
        # k_ = k.to(torch.float64)

        # QK = torch.einsum('bhnd,bhmd->bhnm', q_, k_)
        # mask = torch.triu(torch.ones(QK.size(-2), QK.size(-1)), 1).to(torch.bool).to(QK.device)
        # QK.masked_fill_(mask, float('-inf'))
        
        # # compute rowmax
        # max_vec = QK.max(dim=-1, keepdim=True).values
        
        # QK = QK * (1.0 / (q_.size(-1) ** 0.5))
        # QK = QK * (1.44269504089)
        
        # max_vec = max_vec * (1.44269504089) * (1.0 / (q_.size(-1) ** 0.5))

        # QK = QK - max_vec
        # QK = torch.exp2(QK)
        
        # norm_vec = QK.sum(dim=-1, keepdim=True)
        # max_vec  = max_vec * 0.69314718056
        # norm_vec = torch.log(norm_vec)
        # l_vec   = max_vec + norm_vec
        
        # if (q_.size(-1) == 64):
        #     l_vec = l_vec * -8.0
        # if (q_.size(-1) == 128):
        #     l_vec = l_vec * -11.313708499
        # l_vec = l_vec.to(torch.float)
        ###### AARYAN #####

        diff = torch.max(torch.abs(y_ref - o))
        max_q = torch.max(torch.abs(q))
        max_k = torch.max(torch.abs(k))
        max_v = torch.max(torch.abs(v))
        max_o = torch.max(torch.abs(o))
        mean_q = torch.mean(q)
        mean_k = torch.mean(k)
        mean_v = torch.mean(v)
        mean_o = torch.mean(o)
        std_q = torch.std(q)
        std_k = torch.std(k)
        std_v = torch.std(v)
        std_o = torch.std(o)
        min_q = torch.min(q)
        min_k = torch.min(k)
        min_v = torch.min(v)
        min_o = torch.min(o)
        wandb.log({
            "diff": diff,
            "max_q": max_q,
            "max_k": max_k,
            "max_v": max_v,
            "max_o": max_o,
            "mean_q": mean_q,
            "mean_k": mean_k,
            "mean_v": mean_v,
            "mean_o": mean_o,
            "std_q": std_q,
            "std_k": std_k,
            "std_v": std_v,
            "std_o": std_o,
            "min_q": min_q,
            "min_k": min_k,
            "min_v": min_v,
            "min_o": min_o,
        })

        # check all tensors for nan
        for i, ten in enumerate([q, k, v, o, l_vec, d_vec]):
            if torch.isnan(ten).any():
                print(f"Forwards NaN detected {i}")
                print(torch.isnan(ten).nonzero())
                breakpoint()

        # # save everything to disk 
        # if (diff > 0.1):
        #     fpath = "/scratch/bfs-sim/logged_outputs_forwards/"
        #     if not os.path.exists(fpath):
        #         os.makedirs(fpath)
        #     num_files = len(os.listdir(fpath))
        #     sub_dir = fpath + f"run_{num_files}/"
        #     if not os.path.exists(sub_dir):
        #         os.makedirs(sub_dir)
        #     torch.save(q.cpu(), sub_dir + f"q.pt")
        #     torch.save(k.cpu(), sub_dir + f"k.pt")
        #     torch.save(v.cpu(), sub_dir + f"v.pt")
        #     torch.save(o.cpu(), sub_dir + f"o.pt")
        #     torch.save(l_vec.cpu(), sub_dir + f"l_vec.pt")
        #     torch.save(d_vec.cpu(), sub_dir + f"d_vec.pt")
        #     torch.save(y_ref.cpu(), sub_dir + f"y_ref.pt")
        #     if num_files > 500:
        #         breakpoint()

        ctx.save_for_backward(q, k, v, o, l_vec, d_vec)
        ctx.diff = diff
        return o.to(torch.float32)

    def backward(ctx, grad_output):        
        assert grad_output.shape[3] == 64, "TK train currently supports head dim 64 only"
        diff = ctx.diff 
        
        q, k, v, o, l_vec, d_vec = ctx.saved_tensors

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()
        l_vec = l_vec.contiguous()
        d_vec = d_vec.contiguous()
        grad_o = grad_output.contiguous()

        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        o = o.to(torch.bfloat16)
        grad_o = grad_o.to(torch.bfloat16)
        torch.cuda.synchronize()
        grad_q, grad_k, grad_v = tk.mha_backward(
            q, k, v, o, 
            l_vec, d_vec, 
            grad_o, True
        )   # returns torch.float32 dtypes
        torch.cuda.synchronize()

        # print(f"grad_q: {grad_q.shape}, grad_k: {grad_k.shape}, grad_v: {grad_v.shape}")
        # print(f"grad_q: {grad_q.dtype}, grad_k: {grad_k.dtype}, grad_v: {grad_v.dtype}")

        # check all tensors for nan
        for i, ten in enumerate([q, k, v, o, grad_o, grad_q, grad_k, grad_v, l_vec, d_vec]):
            if torch.isnan(ten).any():
                print(f"Backwards NaN detected {i}")
                print(torch.isnan(ten).nonzero())
                breakpoint()
        
        # if (diff > 0.1):
        #     fpath = "/scratch/bfs-sim/logged_outputs_backwards/"
        #     if not os.path.exists(fpath):
        #         os.makedirs(fpath)
        #     num_files = len(os.listdir(fpath))
        #     sub_dir = fpath + f"run_{num_files}/"
        #     if not os.path.exists(sub_dir):
        #         os.makedirs(sub_dir)
        #     torch.save(q.cpu(), sub_dir + f"q.pt")
        #     torch.save(k.cpu(), sub_dir + f"k.pt")
        #     torch.save(v.cpu(), sub_dir + f"v.pt")
        #     torch.save(o.cpu(), sub_dir + f"o.pt")
        #     torch.save(grad_o.cpu(), sub_dir + f"grad_o.pt")
        #     torch.save(grad_q.cpu(), sub_dir + f"grad_q.pt")
        #     torch.save(grad_k.cpu(), sub_dir + f"grad_k.pt")
        #     torch.save(grad_v.cpu(), sub_dir + f"grad_v.pt")
        #     torch.save(l_vec.cpu(), sub_dir + f"l_vec.pt")
        #     torch.save(d_vec.cpu(), sub_dir + f"d_vec.pt")
        #     if num_files > 500:
        #         breakpoint()

        return grad_q, grad_k, grad_v, None, None, None, None, None, None


class AttentionFunctionBERT(Function):
    def forward(ctx, q, k, v, o, l_vec,  d_vec):        
        assert q.shape[3] == 64, "TK train currently supports head dim 64 only"
        
        l_vec.zero_()
        d_vec.zero_()
        o.zero_()

        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        o = o.to(torch.bfloat16)

        torch.cuda.synchronize()
        tk.mha_forward(q, k, v, o, l_vec, False)
        torch.cuda.synchronize()

        # breakpoint()

        # check all tensors for nan
        for i, ten in enumerate([q, k, v, o, l_vec, d_vec]):
            if torch.isnan(ten).any():
                print(f"Forwards NaN detected {i}")
                print(torch.isnan(ten).nonzero())
                breakpoint()

        ctx.save_for_backward(q, k, v, o, l_vec, d_vec)
        return o.to(torch.float32)

    def backward(ctx, grad_output):        
        assert grad_output.shape[3] == 64, "TK train currently supports head dim 64 only"
        
        q, k, v, o, l_vec, d_vec = ctx.saved_tensors

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()
        l_vec = l_vec.contiguous()
        d_vec = d_vec.contiguous()
        grad_o = grad_output.contiguous()

        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        o = o.to(torch.bfloat16)
        grad_o = grad_o.to(torch.bfloat16)

        # breakpoint()
        torch.cuda.synchronize()
        grad_q, grad_k, grad_v = tk.mha_backward(
            q, k, v, o, 
            l_vec, d_vec, 
            grad_o, False
        )
        torch.cuda.synchronize()
        for i, ten in enumerate([q, k, v, o, grad_o, grad_q, grad_k, grad_v, l_vec, d_vec]):
            if torch.isnan(ten).any():
                print(f"Backwards NaN detected {i}")
                print(torch.isnan(ten).nonzero())
                breakpoint()

        return grad_q, grad_k, grad_v, None, None, None, None, None, None


class CustomAttention(nn.Module):
    def __init__(self, config):
        super(CustomAttention, self).__init__()

        # dimensions
        self.b = config.batch_size
        self.h = config.n_head
        self.n = config.block_size
        self.d = config.n_embd
        self.scale = 1 / (self.d ** 0.5)
        self.is_causal = config.causal 

        print(f"Using CustomAttention -- Causal = {self.is_causal}")

        # weights
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # tensors
        self.outputs = torch.empty((self.b, self.h, self.n, self.d //self. h), dtype=torch.bfloat16, device='cuda')
        self.l_vec   = torch.empty((self.b, self.h, self.n, 1), dtype=torch.float32, device='cuda')
        self.d_vec   = torch.empty((self.b, self.h, self.n, 1), dtype=torch.float32, device='cuda', requires_grad=False)


    def forward(self, x):
        B, T, C = x.size() 
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.d, dim=2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2).contiguous() # (B, nh, T, hs)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2).contiguous() # (B, nh, T, hs)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2).contiguous() # (B, nh, T, hs)

        if self.is_causal:
            attn_fn = AttentionFunction
        else:
            attn_fn = AttentionFunctionBERT

        output = attn_fn.apply(
            q, k, v, self.outputs, self.l_vec, self.d_vec,
        )

        y = output.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_dropout(self.c_proj(y))
        return y


