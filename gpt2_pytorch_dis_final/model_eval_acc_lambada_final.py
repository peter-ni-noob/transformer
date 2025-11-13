import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from dataset import GPT2EvalDataset,GPT2EvalDatasetConfig
import torch.utils.data.dataloader as DataLoader
from lr_scheduler import Anneals_Cosine_LR_Scheduler
import time
import math

import collections
import re

@dataclass
class GPT2Config:
    n_embd=1600
    hidden_dropout=0.0
    attention_dropout=0.0
    batch_size=4
    seq_len=1024
    nlayer=48
    vocab_size=50257
    n_head=25
    
@dataclass
class GPT2OptimizerConfig:
    init_lr=0.0
    wd=1e-1

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
        

def NOddp(md):
    dic=collections.OrderedDict()
    for name,value in md.items():
        new_name=re.sub(r"module.","",name)
        if new_name.find("padding_mask")!=-1:
            continue
        dic[new_name]=value
    return dic



def main(path="/data/Pweight/gpt2xl_weight0.pt",device="cuda:0"):
    model=GPT(GPT2Config())
    model=model.to(device)
    model.eval()

    #加载的权重
    state=torch.load(path)#1105N1epc.pth #gpt2xl_weight0.pt
    ddp_model_dict=state["model_state_dict"]
    
    model.load_state_dict(NOddp(ddp_model_dict),strict=False)

    gptmodelconfig=GPT2Config()
    gptdataset=GPT2EvalDataset(GPT2EvalDatasetConfig(main_path="/data/lambada/",token_num=5153))#token_num=5153
    gptdataloader=DataLoader.DataLoader(gptdataset,batch_size=gptmodelconfig.batch_size,shuffle=False,drop_last=True)    
    
    num_token=0
    corrects=0.0
    print(f"iter_num:{len(gptdataloader)}",flush=True)
    for step,(data,label,mask) in enumerate(gptdataloader):
        with torch.no_grad():
            logits,loss=model(data.to(device))
            label=label.to(device).long()
            mask=mask.to(device)
            num_token+=mask.view(-1).sum()
            outputs = torch.argmax(logits,-1)
            correct = (outputs == label).float()
            correct[(1-mask).bool()]=1
            correct=correct.prod(-1).sum()
            corrects+=correct


        if step%100==0:
            print(f"now step is {step}",flush=True)

    avg_accuracy=corrects/(num_token)
    print(f"num_token:{num_token}",flush=True)
    print(f"number correct:{corrects}",flush=True)

    print(f"avg accuracy:{avg_accuracy}",flush=True)        
            
















if __name__ == "__main__":
    main()