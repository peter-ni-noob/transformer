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

os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

@dataclass
class GPT2DConfig:
    n_embd:int =768
    batch_size: int=8
    seq_len: int =1024
    vocab_size: int=50257

@dataclass
class GPT2DOptimizerConfig:
    init_lr:float=0.01
    wd:float=0.0


iteration=0
mpath="/root/workspace/nexus_gpt2/"
tmp_dir = mpath+"nexusnet/build/Pdata/"
path=mpath+"nexusnet/build/Pdata/"
isdebug=False


def genSavePath(opname=""):
    formated="P_iteration_"+str(iteration)+"_"
    return formated+opname

def genWeight():
    dic=collections.OrderedDict()
    dic["FullyConnected.weight"]=torch.from_numpy(np.load(mpath+"nexusnet/build/netd_weight/Embedding1_embedding.npy"))
    dic["Embedding1.weight"]=torch.from_numpy(np.load(mpath+"nexusnet/build/netd_weight/Embedding1_embedding.npy"))
    dic["Embedding2.weight"]=torch.from_numpy(np.load(mpath+"nexusnet/build/netd_weight/Embedding2_embedding.npy"))
    return dic

ATOL = 1.0e-8
RTOL = 1.0e-5


def load_npy(basename: str) -> np.ndarray:
    return np.load(os.path.join(tmp_dir, basename)).astype(np.float32)


def isSame(a, b):
    # print(a)
    # print(b)
    # int_arr1 = np.asarray(a.view(np.int32))
    # int_arr2 = np.asarray((b).view(np.int32))
    np.testing.assert_allclose(a, b, atol=ATOL, rtol=RTOL)
    print("------------------")
    print("mean diff ", np.mean(np.abs(a - b)))
    # print("max ulpdiff ", np.max(np.abs(int_arr1 - int_arr2)))
    # index=np.argmax(np.abs(int_arr1 - int_arr2))
    # print("isSame--max ulpdiff index ",np.argmax(np.abs(int_arr1 - int_arr2)))
    # ulp相差最大的两个数的整数表示
    # print("maxdiff ints ",int_arr1.ravel()[index], int_arr2.ravel()[index],(int_arr1.ravel()[index] - int_arr2.ravel()[index]))
    # ulp相差最大的两个数的浮点数表示
    # print("maxdiff floats ",a.ravel()[index], b.ravel()[index],(a.ravel()[index] - b.ravel()[index]))
    print("max diff ", np.max(np.abs(a - b)))
    # print("max diff index ",np.argmax(np.abs(a - b)))
    print("max rele diff", np.max(np.abs(a - b) / np.abs(b)))
    print("------------------")


def save_npy(basename: str, x: Union[torch.Tensor, np.ndarray]):
    global isdebug
    if(isdebug==False):
        return
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy().astype(np.float32)
    assert isinstance(x, np.ndarray)
    np.save(os.path.join(tmp_dir, basename), x)

def hook_forward_function(module, module_input, module_output):
    # pass
    # print(len(module_input))
    save_npy(genSavePath("back_"+module.name+"_input_x.npy"),module_input[0])


def hook_backward_function(module,module_input_grad,module_output_grad):
    # 1/0
    save_npy(genSavePath("back_"+module.name+"_input_g.npy"),module_output_grad[0])
    if(module.name.find("Embedding")==-1):
        save_npy(genSavePath("back_"+module.name+"_input_w.npy"),module.weight)
    # print(module)
    # 1/0
    # print(len(module_input_grad))
    # print(module_output_grad)


class GPT2D(nn.Module):
    def __init__(self,config:GPT2DConfig):
        super().__init__()
        self.config=config
        self.Embedding1 = nn.Embedding(config.vocab_size,config.n_embd)
        self.Embedding1.name="Embedding1"
        self.Embedding2 = nn.Embedding(config.seq_len,config.n_embd)
        self.Embedding2.name="Embedding2"

        self.GELU    = nn.GELU(approximate='tanh')
        self.FullyConnected  = nn.Linear(config.n_embd, config.vocab_size,False)
        self.FullyConnected.name="FullyConnected"


    def forward(self,data,label):
        data=data.long()
        label=label.long()
        pos = torch.arange(0,self.config.seq_len,dtype=torch.long,device=data.device)
        pos=pos.repeat(self.config.batch_size,1)
        save_npy(genSavePath("Embedding1_input_0.npy"),data)
        save_npy(genSavePath("Embedding1_input_1.npy"),self.Embedding1.weight)
        wte=self.Embedding1(data)
        save_npy(genSavePath("Embedding1_output_0.npy"),wte)
        wpe=self.Embedding2(pos)
        save_npy(genSavePath("Embedding2_input_0.npy"),pos)
        save_npy(genSavePath("Embedding2_input_1.npy"),self.Embedding2.weight)
        save_npy(genSavePath("Embedding2_output_0.npy"),wpe)
        save_npy(genSavePath("Sum_input_0.npy"),wte)
        save_npy(genSavePath("Sum_input_1.npy"),wpe)
        wtpes=wte+wpe
        save_npy(genSavePath("Sum_output_0.npy"),wtpes)
        block=self.GELU(wtpes)
        save_npy(genSavePath("GELU_input_0.npy"),wtpes)
        save_npy(genSavePath("GELU_output_0.npy"),block)
        linear=self.FullyConnected(block)
        save_npy(genSavePath("FullyConnected_input_0.npy"),block)
        save_npy(genSavePath("FullyConnected_input_1.npy"),self.FullyConnected.weight)
        save_npy(genSavePath("FullyConnected_output_0.npy"),linear)
        logits=torch.reshape(linear,[self.config.batch_size*self.config.seq_len,self.config.vocab_size])
        save_npy(genSavePath("Reshape1_input_0.npy"),linear)
        save_npy(genSavePath("Reshape1_output_0.npy"),logits)
        labels=torch.reshape(label,[self.config.batch_size*self.config.seq_len])
        save_npy(genSavePath("Reshape2_input_0.npy"),label)
        save_npy(genSavePath("Reshape2_output_0.npy"),labels)
        loss=F.cross_entropy(logits,labels)
        save_npy(genSavePath("CEloss_input_0.npy"),logits)
        save_npy(genSavePath("CEloss_input_1.npy"),labels)
        save_npy(genSavePath("CEloss_output_0.npy"),loss)



        
        return loss

def main():
    gpt2Dmodelconfig=GPT2DConfig()
    model=GPT2D(gpt2Dmodelconfig)
    device = "cuda:0"

    weight=genWeight()
    model.load_state_dict(weight)
    model=model.to(device)
    model.FullyConnected.register_full_backward_hook(hook_backward_function)
    model.FullyConnected.register_forward_hook(hook_forward_function)
    model.Embedding1.register_full_backward_hook(hook_backward_function)
    model.Embedding2.register_forward_hook(hook_forward_function)



    Opt_config=GPT2DOptimizerConfig()
    optimizer = torch.optim.AdamW(model.parameters(),lr=Opt_config.init_lr,weight_decay=Opt_config.wd,foreach=False)
    gptdataset=GPT2dataset(GPT2datasetConfig())
    sampler=GPTDistributed_Sampler(gptdataset)
    gptdataloader=DataLoader.DataLoader(gptdataset,batch_size=gpt2Dmodelconfig.batch_size,shuffle=False,sampler=sampler)
    for step,(data,label) in enumerate(gptdataloader):
        optimizer.zero_grad()
        global iteration
        iteration=step+1
        loss=model(data.to(device),label.to(device))
        ans=loss.detach().item()
        print("loss: %.8f"%ans,"iter:",step+1)
        loss.backward()
        save_npy(genSavePath("back_Embedding1_output_1.npy"),model.Embedding1.weight.grad)
        save_npy(genSavePath("back_Embedding2_output_1.npy"),model.Embedding2.weight.grad)
        save_npy(genSavePath("back_FullyConnected_output_1.npy"),model.FullyConnected.weight.grad)

        optimizer.step()
        ans=optimizer.state_dict()
        print(ans)
        if(step+1==13):
            break


if __name__ == "__main__":
    main()