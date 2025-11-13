import torch 
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

import os
import sys
import time


from typing import Union

tmp_dir = "/root/workspace/nexus20240914/nexusnet/build/"


def save_npy(basename: str, x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy().astype(np.float32)
    assert isinstance(x, np.ndarray)
    np.save(os.path.join(tmp_dir, basename), x)

device="cuda:2"



x=torch.randn([8,1024,50257],device=device,dtype=torch.float32,requires_grad=True)
save_npy("x.npy",x.view(-1,x.size(-1)))
y=torch.randint(0,50256,[8,1024],device=device,dtype=torch.float32,requires_grad=True)
save_npy("y.npy",y.view(-1).type(torch.float32))
y=y.type(torch.long)
# f=nn.GELU("tanh").to(device)
t_begin=time.time()
for i in range(200):
    l=F.cross_entropy(x.view(-1,x.size(-1)),y.view(-1))
    l.backward()
t_end=time.time()
dlta_t=t_end-t_begin
print(dlta_t)
