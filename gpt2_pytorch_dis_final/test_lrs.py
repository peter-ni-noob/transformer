import torch
import torch.distributed
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import torch.utils
from dataset import GPT2dataset
from dataset import GPT2datasetConfig
from dataset import GPTDistributed_Sampler


import torch.utils.data.dataloader as DataLoader
from lr_scheduler import Anneals_Cosine_LR_Scheduler
import time
import os
import sys
from contextlib import nullcontext
import collections
import re
import numpy as np
from typing import Union

ATOL = 1.0e-8
RTOL = 1.0e-5

def isSame(a, b):
    # print(a)
    # print(b)
    # int_arr1 = np.asarray(a.view(np.int32))
    # int_arr2 = np.asarray((b).view(np.int32))
    try:
        np.testing.assert_allclose(a, b, atol=ATOL, rtol=RTOL)
    except Exception as e:
        dlta=np.mean(np.abs(a - b))

        return False,dlta
    # print("------------------")
    dlta=np.mean(np.abs(a - b))
    if(dlta!=0.0):
        return False,dlta
    return True,dlta


def save_npy_raw(basename: str, x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, torch.Tensor):
        x = x.contiguous().cpu().detach().numpy().astype(np.float64)
    assert isinstance(x, np.ndarray)
    np.save(basename, x)


def compare():
    Ppath="/root/workspace/nexus_gpt2/nexusnet/build/Pdata0"
    Npath="/root/workspace/nexus_gpt2/nexusnet/build/Ndata0"
    file_names = [f for f in os.listdir(Ppath) if os.path.isfile(os.path.join(Ppath, f))]
    file_names.sort()
    for i in range(len(file_names)):
        ndata=np.load(os.path.join(Npath,file_names[i]))
        pdata=np.load(os.path.join(Ppath,file_names[i]))
        # if(file_names[i].find("embedding")!=-1):
        #     pass
        ans=isSame(ndata,pdata)
        if(ans[0]==False):
            print(file_names[i],ans[1],flush=True)

def pgen():
    scheduler=Anneals_Cosine_LR_Scheduler()
    for i in range(100):
        lr=scheduler.step_lr()
        a=torch.tensor([lr],dtype=torch.float64,device="cpu")
        save_npy_raw("/root/workspace/nexus_gpt2/nexusnet/build/Pdata0/"+str(i)+".npy",a)

def ngen():
    os.system("./build/test/test_lrs")

if __name__ == "__main__":
    pgen()
    ngen()
    compare()