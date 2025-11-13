import torch 
import torch.nn as nn
import numpy as np

import os
import sys
import time


from typing import Union
import nvtx

tmp_dir = "/root/workspace/nexus20240914/nexusnet/build"


def save_npy(basename: str, x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy().astype(np.float32)
    assert isinstance(x, np.ndarray)
    np.save(os.path.join(tmp_dir, basename), x)

device="cuda:2"

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=768,num_heads=12,dropout=0.1,batch_first=True)

        self.register_buffer("padding_mask",torch.zeros(8,1024))
        self.register_buffer("attention_mask",nn.Transformer.generate_square_subsequent_mask(1024))
        # print(self.attention_mask)

    def forward(self,x):
        x,_=self.attn(x,x,x,self.padding_mask,attn_mask=self.attention_mask,is_causal=True)
        return x


x=torch.randn([8,1024,768],device=device,dtype=torch.float32,requires_grad=True)
save_npy("x.npy",x)
f=Attention().to(device)
t_begin=time.time()
for i in range(200*12):
    with nvtx.annotate(message="loop",color="red"):
    
        y=f(x)
        y.mean().backward()
t_end=time.time()
dlta_t=t_end-t_begin
print(dlta_t)
