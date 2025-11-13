import torch 
import torch.nn as nn
import numpy as np

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



x=torch.randn([8,1024,768],device=device,dtype=torch.float32,requires_grad=True)
save_npy("x.npy",x)
f=nn.Linear(768,768*4).to(device)
t_begin=time.time()
for i in range(200*(12*2+1)):
    y=f(x)
    y.mean().backward()
t_end=time.time()
dlta_t=t_end-t_begin
print(dlta_t)
