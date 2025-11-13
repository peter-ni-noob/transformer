import torch
import torch.nn as nn
from torch import autograd
import numpy as np
from typing import Union
import os
import re

tmp_dir="/root/workspace/nexus_gpt2/nexusnet/build/Pdata0/"
isdebug=True
def load_npy(basename: str) -> np.ndarray:
    return np.load(os.path.join(tmp_dir, basename)).astype(np.float32)

def save_npy(basename: str, x: Union[torch.Tensor, np.ndarray]):
    global isdebug
    ans=re.compile(r"\d+").search(basename)
    1/0
    if (ans is not None) and ans.group()!="47":
        
        return
    if(isdebug==False):
        return
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy().astype(np.float32)
    assert isinstance(x, np.ndarray)
    np.save(os.path.join(tmp_dir, basename), x)

def hook_backward_function(module,module_input_grad,module_output_grad):
    save_npy("back_"+module.name+str(module.cnt_b)+"_gdebugop.npy",module_input_grad[0])
    module.cnt_b+=1

    pass


class debugOP_(nn.Module):
    
    def __init__(self,name):
        super().__init__()
        self.cnt=0
        self.cnt_b=0
        self.name=name
        self.register_full_backward_hook(hook_backward_function)

    def forward(self,x):
        save_npy(self.name+str(self.cnt)+"_debugop.npy",x)
        self.cnt+=1
        return x

class debugOP(autograd.Function):
    cnt=0
    @staticmethod
    def forward(ctx,input,name):
        cntt=debugOP.cnt
        ctx.constant=(cntt,name)
        save_npy(name+str(cntt)+"_debugop.npy",input)
        debugOP.cnt+=1
        return input
    
    # @staticmethod
    # def setup_context(ctx,inputs,output):
    #     input,name=inputs
    #     ctx.constant=(name,debugOP.cnt)
    #     pass

    @staticmethod
    def backward(ctx, grad_output):
        cntt,name=ctx.constant
        save_npy("back_"+name+str(cntt)+"_gdebugop.npy",grad_output)
        return grad_output,None




# if __name__ == "__main__":
#     x = torch.rand([10,10], dtype=torch.float32).to("cuda:0").requires_grad_()
#     net=debugOP().apply
#     y=net(x,"annt")
#     z=net(y,"anntx")
#     z.mean().backward()
#     save_npy("xgrad",x.grad)
#     print(x.grad)

