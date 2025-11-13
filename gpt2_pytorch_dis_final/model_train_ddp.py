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
    print(f"using device: {device}")



@dataclass
class GPT2Config:
    n_embd=768
    hidden_dropout=0.1
    attention_dropout=0.1
    batch_size=8
    seq_len=1024
    nlayer=12
    vocab_size=50257
    n_head=12    
    
@dataclass
class GPT2OptimizerConfig:
    init_lr=0.00015
    wd=1e-2

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
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
        self.dropout=nn.Dropout(config.hidden_dropout)
        self.register_buffer("padding_mask",torch.zeros(config.batch_size,config.seq_len))
        self.register_buffer("attention_mask",nn.Transformer.generate_square_subsequent_mask(config.seq_len))
        # print(self.attention_mask)

    def forward(self,x):
        x,_=self.attn(x,x,x,self.padding_mask,attn_mask=self.attention_mask,is_causal=True)
        x=self.dropout(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(self,config:GPT2Config):
        super().__init__()
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention=Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
    def forward(self,x):
        x=x+self.attention(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):

    def __init__(self,config:GPT2Config):
        super().__init__()
        self.config=config
        self.transformer =nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.seq_len,config.n_embd),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.nlayer)]),
            ln_f = nn.LayerNorm(config.n_embd),
            hdropout = nn.Dropout(config.hidden_dropout)
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.transformer.wte.weight =self.lm_head.weight

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
        x = tok_emb + pos_emb
        x=self.transformer.hdropout(x)
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if label is not None:
            label=label.long()
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),label.view(-1))
        return logits,loss
        





def main():
    model=GPT(GPT2Config())
    device = "cuda:2"
    model=model.to(device)

    if ddp:
        model= DDP(model,device_ids=[ddp_local_rank],output_device=ddp_local_rank)
    model.train(True)

    Opt_config=GPT2OptimizerConfig()
    optimizer = torch.optim.AdamW(model.parameters(),lr=Opt_config.init_lr,weight_decay=Opt_config.wd)
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
        
        

    
    # print(len(gptdataloader))
    loss_acc=0.0
    t_begin=time.time()
    for step,(data,label) in enumerate(gptdataloader):
        optimizer.zero_grad()
        _,loss=model(data.to(device),label.to(device))
        loss_acc+=loss.detach().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        lr=scheduler.step_lr()
        for param_group in optimizer.param_groups:
            param_group["lr"]=lr
        optimizer.step()
        if(step%100==0 and ddp_local_rank==0):
            t_end=time.time()
            dlta_t=t_end-t_begin
            if(step==0):
                print(f"iteration: {step} | loss: {loss_acc} | elapsed time(s): {dlta_t} | throughput(tokens/s): {1.0*gptmodelconfig.batch_size*gptmodelconfig.seq_len*ddp_world_size/dlta_t}")
            else:
                print(f"iteration: {step} | loss: {loss_acc/100} | elapsed time(s): {dlta_t} | throughput(tokens/s): {100*gptmodelconfig.batch_size*gptmodelconfig.seq_len*ddp_world_size/dlta_t}")
            loss_acc=0.0
            t_begin=t_end

        if(step%20000==0 and ddp_local_rank==0):
            state={
                "step":step,
                "optimizer":optimizer.state_dict(),
                "model_state_dict":model.state_dict(),
                "scheduler" : scheduler
            }
            torch.save(state,"%d.pth"%step)










if __name__ == "__main__":
    main()

if ddp:
    destroy_process_group()