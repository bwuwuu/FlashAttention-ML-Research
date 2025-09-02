import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

def attention(Q, K, V):
    score = (Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(K.size(-1))))
    score = F.softmax(score, dim=-1)
    out = score @ V
    return out

batch_size = 4 # 1, 4, 16, 64
n_head = 8
seq_len = 64 # 128
d_model = 64

flash_attention = load(name='flash', sources=['flash.cpp', 'flash.cu'], extra_cuda_cflags=['-O2']) # verbose=True

Q = torch.randn(batch_size, n_head, seq_len, d_model).cuda()
K = torch.randn(batch_size, n_head, seq_len, d_model).cuda()
V = torch.randn(batch_size, n_head, seq_len, d_model).cuda()

print('------------------------- profiling standard attention -------------------------')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    standard_att_out = attention(Q, K, V)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=15))

print('-------------------------- profiling flash attention --------------------------')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    flash_att_out = flash_attention.flash_forward(Q, K, V)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=15))

print('sanity check:', torch.allclose(standard_att_out, flash_att_out, rtol=0, atol=1e-04))

