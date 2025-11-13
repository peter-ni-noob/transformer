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
import shutil
from typing import Union
import numpy as np
import collections



from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import re



mpath="/root/workspace/nexus_gpt2/"

tmp_dir = mpath+"nexusnet/build/Pdata/"
tmp_dir_n=mpath+"nexusnet/build/Ndata/"

Ppath=mpath+"nexusnet/build/Pdata/"
Npath=mpath+"nexusnet/build/Ndata/"

ATOL = 1.0e-8
RTOL = 1.0e-5

errorName=[]

def load_npy(basename: str) -> np.ndarray:
    return np.load(os.path.join(tmp_dir, basename)).astype(np.float32)
    
def load_npy_n(basename: str) -> np.ndarray:
    return np.load(os.path.join(tmp_dir_n, basename)).astype(np.float32)

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

    # print("max ulpdiff ", np.max(np.abs(int_arr1 - int_arr2)))
    # index=np.argmax(np.abs(int_arr1 - int_arr2))
    # print("isSame--max ulpdiff index ",np.argmax(np.abs(int_arr1 - int_arr2)))
    # ulp相差最大的两个数的整数表示
    # print("maxdiff ints ",int_arr1.ravel()[index], int_arr2.ravel()[index],(int_arr1.ravel()[index] - int_arr2.ravel()[index]))
    # ulp相差最大的两个数的浮点数表示
    # print("maxdiff floats ",a.ravel()[index], b.ravel()[index],(a.ravel()[index] - b.ravel()[index]))
    # print("max diff ", np.max(np.abs(a - b)))
    # # print("max diff index ",np.argmax(np.abs(a - b)))
    # print("max rele diff", np.max(np.abs(a - b) / np.abs(b)))
    # print("------------------")


def save_npy(basename: str, x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy().astype(np.float32)
    assert isinstance(x, np.ndarray)
    np.save(os.path.join(tmp_dir, basename), x)



def main():


    dir_path = Ppath
    file_names = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    for i in range(len(file_names)):
        file_names[i]=file_names[i][1:]
        ndata=np.load(os.path.join(Npath,"N"+file_names[i]))
        pdata=np.load(os.path.join(Ppath,"P"+file_names[i]))
        ans=isSame(ndata,pdata)
        if(ans[0]==False):
            errorName.append((file_names[i],ans[1]))
    errorName.sort(key=lambda x: x[0])
    for i in errorName:
        print(i)


def compare_b():
    error=[]
    dir_path = Ppath
    file_names = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    file_names.sort()
    for i in range(len(file_names)):
        # ans=re.compile(r"back_attn\d*_output_x").search(file_names[i])
        ans=re.compile(r"attn\d*_input_0").search(file_names[i])
        if(ans is not None):
            continue
        ndata=np.load(os.path.join(Npath,file_names[i]))
        pdata=np.load(os.path.join(Ppath,file_names[i]))
        # if(file_names[i].find("embedding")!=-1):
        #     pass
        ans=isSame(ndata,pdata)
        if(ans[0]==False):
            print(file_names[i],ans[1],flush=True)

        # print(file_names[i],flush=True)
        pass



def cp_opt():
    # a=load_npy_n("1_Embedding1_weight_after.npy")
    # b=load_npy_n("N_iteration_2_Embedding1_input_1.npy")
    # ans=isSame(a,b)
    # print(ans)

    a=load_npy_n("N_iteration_1_Embedding1_input_1.npy")
    b=load_npy("P_iteration_1_Embedding1_input_1.npy")
    ans=isSame(a,b)
    
    print(ans)

    


if __name__ == "__main__":
    compare_b()