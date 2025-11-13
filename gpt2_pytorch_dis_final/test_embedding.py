import torch 
import torch.nn as nn
import numpy as np

import os
import sys
import time


from typing import Union

tmp_dir = "/root/workspace/neuxs0807/nexusnet/build"


def save_npy(basename: str, x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy().astype(np.float32)
    assert isinstance(x, np.ndarray)
    np.save(os.path.join(tmp_dir, basename), x)

device="cuda:2"



x=torch.randint(0,50256,[8,1024],device=device,dtype=torch.float32,requires_grad=True)
save_npy("x.npy",x.type(torch.float32))
x=x.long()
f=nn.Embedding(50257,768).to(device)
t_begin=time.time()
for i in range(200*(2)):
    y=f(x)
    y.mean().backward()
t_end=time.time()
dlta_t=t_end-t_begin
print(dlta_t)
