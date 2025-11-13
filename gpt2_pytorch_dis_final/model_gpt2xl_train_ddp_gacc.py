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



from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


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
    print(f"using device: {device}",flush=True)



@dataclass
class GPT2Config:
    n_embd=1600
    hidden_dropout=0.0
    attention_dropout=0.0
    batch_size=1
    seq_len=1024
    nlayer=48
    vocab_size=50257
    n_head=25
    accumulation_steps=1
    
@dataclass
class GPT2OptimizerConfig:
    init_lr=0.0
    wd=1e-1#1e-2

iteration=1
mpath="/root/workspace/nexus_gpt2/"
tmp_dir = mpath+"nexusnet/build/Pdata/"
path=mpath+"nexusnet/build/Pdata/"
isdebug=False
isdebug_back=True
def genSavePath(opname=""):
    formated="P_iteration_"+str(iteration)+"_"
    return formated+opname

def genSavePath_b(opname=""):
    formated="iteration_"+str(iteration)+"_"
    return formated+opname

def load_npy(basename: str) -> np.ndarray:
    return np.load(os.path.join(tmp_dir, basename)).astype(np.float32)

def save_npy(basename: str, x: Union[torch.Tensor, np.ndarray]):
    global isdebug
    if(isdebug==False):
        return
    if isinstance(x, torch.Tensor):
        x = x.contiguous().cpu().detach().numpy().astype(np.float32)
    assert isinstance(x, np.ndarray)
    np.save(os.path.join(tmp_dir, basename), x)


def save_npy_b(basename: str, x: Union[torch.Tensor, np.ndarray]):
    global isdebug
    if(isdebug_back==False):
        return
    if isinstance(x, torch.Tensor):
        x = x.contiguous().cpu().detach().numpy().astype(np.float32)
    assert isinstance(x, np.ndarray)
    np.save(os.path.join(tmp_dir, basename), x)

def hook_forward_function(module, module_input, module_output):
    # pass
    # print(module)
    # print(len(module_input))
    if(hasattr(module,"name")):
        if(module.opname=="sum"):
            save_npy(genSavePath(module.name+"_input_0.npy"),module_input[0])
            save_npy(genSavePath(module.name+"_input_1.npy"),module_input[1])
            save_npy(genSavePath(module.name+"_output_0.npy"),module_output[0])

            pass
        elif (module.opname=="embedding"):
            save_npy(genSavePath(module.name+"_input_0.npy"),module_input[0])
            save_npy(genSavePath(module.name+"_output_0.npy"),module_output)
            save_npy(genSavePath(module.name+"_input_1.npy"),module.weight)
        elif (module.opname=="layernorm"):
            save_npy(genSavePath(module.name+"_input_x.npy"),module_input[0])
            save_npy(genSavePath(module.name+"_output_0.npy"),module_output)
        elif (module.opname=="dropout"):
            save_npy(genSavePath(module.name+"_input_0.npy"),module_input[0])
            save_npy(genSavePath(module.name+"_output_0.npy"),module_output)
        elif(module.opname=="CrossEntropy"):
            save_npy(genSavePath(module.name+"_input_0.npy"),module_input[0])
            save_npy(genSavePath(module.name+"_input_1.npy"),module_input[1])
            save_npy(genSavePath(module.name+"_output_0.npy"),module_output)

        elif(module.opname=="mha" and isdebug_back):
            save_npy_b(genSavePath_b(module.name+"_input_0.npy"),module_input[0])    

    # save_npy(genSavePath("back_"+module.name+"_input_x.npy"),module_input[0])
    pass
#grad 按拓扑排序
def hook_backward_function(module,module_input_grad,module_output_grad):

    if(hasattr(module,"name")):
        if(module.opname=="layernorm"):
            save_npy_b(genSavePath_b(module.name+"_gamma.npy"),module.weight.grad)
            save_npy_b(genSavePath_b(module.name+"_beta.npy"),module.bias.grad)
        elif(module.opname=="mha"):
            # 1/0
            save_npy_b(genSavePath_b("back_"+module.name+"_output_x.npy"),module_input_grad[0]+module_input_grad[1]+module_input_grad[2])
            save_npy_b(genSavePath_b("back_"+module.name+"_input_g.npy"),module_output_grad[0])
            
            save_npy_b(genSavePath_b(module.name+"in_proj_bias.npy"),module.in_proj_bias.grad)
            save_npy_b(genSavePath_b(module.name+"in_proj_weight.npy"),module.in_proj_weight.grad)
            save_npy_b(genSavePath_b(module.name+"out_proj_bias.npy"),module.out_proj.bias.grad)
            save_npy_b(genSavePath_b(module.name+"out_proj_weight.npy"),module.out_proj.weight.grad)
        elif(module.opname=="linear"):
            save_npy_b(genSavePath_b(module.name+"_weight.npy"),module.weight.grad)
            save_npy_b(genSavePath_b(module.name+"_bias.npy"),module.bias.grad)
        # embedding 从hook得不到grad
        # elif(module.opname=="embedding" and module.name=="wpe"):
        #     save_npy_b(genSavePath_b(module.name+"_embedding.npy"),module.weight.grad)
            

    pass


class ADD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y):
        return x+y

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,logits,label):
        return F.cross_entropy(logits.view(-1,logits.size(-1)),label.view(-1))


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_fc.name="c_fc"
        self.c_fc.opname="linear"
        self.c_fc.register_full_backward_hook(hook_backward_function)

        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.name="c_proj"
        self.c_proj.opname="linear"
        self.c_proj.register_full_backward_hook(hook_backward_function)
        self.dropout =nn.Dropout(config.hidden_dropout)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=config.n_embd,num_heads=config.n_head,dropout=config.attention_dropout,batch_first=True)
        self.attn.name="attn"
        self.attn.opname="mha"
        # self.attn.out_proj.weight
        self.attn.register_full_backward_hook(hook_backward_function)
        self.attn.register_forward_hook(hook_forward_function)#会导致进入不同分支
        self.dropout=nn.Dropout(config.hidden_dropout)
        self.dropout.name="attn_dropout_post"
        self.dropout.opname="dropout"
        self.dropout.register_forward_hook(hook_forward_function)
        # self.register_buffer("padding_mask",torch.zeros(config.batch_size,config.seq_len))
        self.register_buffer("attention_mask",nn.Transformer.generate_square_subsequent_mask(config.seq_len))
        # print(self.attention_mask)

    def forward(self,x):
        x,_=self.attn(x,x,x,attn_mask=self.attention_mask,is_causal=True)
        x=self.dropout(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(self,config:GPT2Config):
        super().__init__()
        self.mlp = MLP(config)
        self.mlp.c_fc.name+="ffw"
        self.mlp.c_proj.name+="ffw"
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention=Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ln_1.name="ln1"
        self.ln_1.opname="layernorm"
        self.ln_2.name="ln2"
        self.ln_2.opname="layernorm"
        self.ln_1.register_forward_hook(hook_forward_function)
        self.ln_1.register_full_backward_hook(hook_backward_function)
        self.ln_2.register_full_backward_hook(hook_backward_function)
        
        
    def forward(self,x):
        x=x+self.attention(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self,config:GPT2Config):
        super().__init__()
        self.config=config
        h_list=[]
        for i in range(config.nlayer):
            h_list.append(TransformerBlock(config))
            h_list[i].ln_1.name+=str(i)
            h_list[i].ln_2.name+=str(i)
            h_list[i].attention.dropout.name+=str(i)
            h_list[i].attention.attn.name+=str(i)
            # h_list[i].attention.attn.out_proj.weight.name=h_list[i].attention.attn.name

            h_list[i].mlp.c_fc.name+=str(i)
            h_list[i].mlp.c_proj.name+=str(i)


        self.transformer =nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.seq_len,config.n_embd),
            h = nn.ModuleList(h_list),
            ln_f = nn.LayerNorm(config.n_embd),
            hdropout = nn.Dropout(config.hidden_dropout)
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.transformer.wte.weight =self.lm_head.weight
        self.celoss=CrossEntropy()
        self.celoss.name="loss"
        self.celoss.opname="CrossEntropy"
        self.celoss.register_forward_hook(hook_forward_function)
        self.transformer.ln_f.name="ln_final"
        self.transformer.ln_f.opname="layernorm"
        self.transformer.ln_f.register_full_backward_hook(hook_backward_function)


        self.transformer.wte.name="wte"
        self.transformer.wte.opname="embedding"
        self.transformer.wpe.name="wpe"
        self.transformer.wpe.opname="embedding"
        self.transformer.wte.register_forward_hook(hook_forward_function)
        self.transformer.wpe.register_forward_hook(hook_forward_function)
        self.transformer.wte.register_full_backward_hook(hook_backward_function)
        self.transformer.wpe.register_full_backward_hook(hook_backward_function)
        self.add=ADD()
        self.add.name="wtpes"
        self.add.opname="sum"
        self.add.register_forward_hook(hook_forward_function)



        self.apply(self._init_weights)

    def _init_weights(self,module):
            std = 0.02
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                torch.nn.init.normal_(module.in_proj_weight,mean=0,std=std)
                torch.nn.init.normal_(module.out_proj.weight,mean=0,std=std)

            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,input,label=None):
        input=input.long()
        pos = torch.arange(0,self.config.seq_len,dtype=torch.long,device=input.device)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(input) # token embeddings of shape (B, T, n_embd)
        x = self.add(tok_emb,pos_emb)
        x=self.transformer.hdropout(x)
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if label is not None:
            label=label.long()
            
            loss = self.celoss(logits,label)
        return logits,loss
        

def save_dic_toCPU(md):
    dic=collections.OrderedDict()
    for name,value in md.items():
        dic[name]=value.to("cpu")
    return dic

def load_dic_toGPU(md,device):
    for name,value in md:
        md[name]=value.to(device)
    return md

def NOddp(md):
    dic=collections.OrderedDict()
    for name,value in md.items():
        new_name=re.sub(r"module.","",name)
        if new_name.find("padding_mask")!=-1:
            continue
        dic[new_name]=value
    return dic

def main():
    model=GPT(GPT2Config())
    # device = "cuda:0"
    model=model.to(device)

    state_dict0=torch.load("/data/Pweight/0_gpt2xl.pth") # /data/Pweight/0_gpt2xl.pth  gpt2xlHugginFace_pweight.pth
    state_dict0=NOddp(state_dict0["model_state_dict"])
    model.load_state_dict(state_dict0,strict=False)
    torch.cuda.manual_seed_all(5067878658844510)


    if ddp:
        model= DDP(model,device_ids=[ddp_local_rank],output_device=ddp_local_rank)
    model.train(True)
    # model.name="gpt2"
    # model.opname="none"
    model.register_forward_hook(hook_forward_function)


    # for i in model.modules():
    #     print(i)

    Opt_config=GPT2OptimizerConfig()
    optimizer = torch.optim.AdamW(model.parameters(),lr=Opt_config.init_lr,weight_decay=Opt_config.wd,foreach=False)
    scheduler=Anneals_Cosine_LR_Scheduler()

    


    gptmodelconfig=GPT2Config()
    gptdataset=GPT2dataset(GPT2datasetConfig())
    # import torch.utils.data.distributed
    # sampler=torch.utils.data.distributed.DistributedSampler(gptdataset, num_replicas=None, rank=None, shuffle=False, seed=0, drop_last=True)
    sampler=GPTDistributed_Sampler(gptdataset)
    



    gptdataloader=DataLoader.DataLoader(gptdataset,batch_size=gptmodelconfig.batch_size,shuffle=False,sampler=sampler)
    # obj=iter(gptdataloader)
    # for i in sampler:
    #     x=next(obj)
    #     print(i,gptdataset[i][0],x[0])
        
        
    accumulation_steps=gptmodelconfig.accumulation_steps
    globalBatchsize=gptmodelconfig.batch_size*gptmodelconfig.accumulation_steps
    
    # print(len(gptdataloader))
    loss_acc=0.0
    t_begin=time.time()
    real_step=0
    for step,(data,label) in enumerate(gptdataloader):
        
        mcontext = model.no_sync if ddp and (step+1) % accumulation_steps != 0 else nullcontext
        with mcontext():
            _,loss=model(data.to(device),label.to(device))
            loss_acc+=loss.detach().item()
            loss=loss/accumulation_steps
            loss.backward()
            save_npy_b(genSavePath_b(model.transformer.wpe.name+"_embedding.npy"),model.transformer.wpe.weight.grad)
            save_npy_b(genSavePath_b(model.transformer.wte.name+"_embedding.npy"),model.transformer.wte.weight.grad)

        if (step+1)%accumulation_steps ==0:
            # print(step)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            # loss_acc+=accumulation_steps*(loss.detach().item())
            loss_acc =loss_acc/accumulation_steps
            
            lr=scheduler.step_lr()
            for param_group in optimizer.param_groups:
                param_group["lr"]=lr

            optimizer.step()

            optimizer.zero_grad()
            if(real_step%1==0 and ddp_local_rank==0):
                t_end=time.time()
                dlta_t=t_end-t_begin
                if(real_step==0):
                    print(f"iteration: {real_step} | loss: {loss_acc} | elapsed time(s): {dlta_t} | throughput(tokens/s): {1.0*globalBatchsize*gptmodelconfig.seq_len*ddp_world_size/dlta_t}")
                else:
                    print(f"iteration: {real_step} | loss: {loss_acc/1} | elapsed time(s): {dlta_t} | throughput(tokens/s): {1*globalBatchsize*gptmodelconfig.seq_len*ddp_world_size/dlta_t}")
                loss_acc=0.0
                t_begin=t_end

            if(real_step%1000==0 and ddp_local_rank==0):
                # state={
                #     "step":real_step,
                #     # "optimizer":optimizer.state_dict(),
                #     "model_state_dict":save_dic_toCPU(model.state_dict()),
                #     "scheduler" : scheduler
                # }
                # torch.save(state,"%d_gpt2xl.pth"%real_step)
                pass
            real_step+=1
            global iteration
            iteration+=1
            if(real_step==1):
                return

        
        











if __name__ == "__main__":
    main()

if ddp:
    destroy_process_group()