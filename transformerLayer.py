import torch
import torch.distributed
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from hpConfig import TransformerConfig

from tpLayer import ParallelAttention,ParallelMLP

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,logits,label):
        return F.cross_entropy(logits.view(-1,logits.size(-1)),label.view(-1))
    

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd, config.intermidiate_size)
        self.c_fc.name="c_fc"
        self.c_fc.opname="linear"


        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(config.intermidiate_size, config.n_embd)
        self.c_proj.name="c_proj"
        self.c_proj.opname="linear"

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
        self.dropout=nn.Dropout(config.hidden_dropout)
        self.dropout.name="attn_dropout_post"
        self.dropout.opname="dropout"
        self.register_buffer("attention_mask",nn.Transformer.generate_square_subsequent_mask(config.seq_len))

    def forward(self,x):
        x,_=self.attn(x,x,x,attn_mask=self.attention_mask,is_causal=True)
        x=self.dropout(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention=Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.mlp.c_fc.name+="ffw"
        self.mlp.c_proj.name+="ffw"
        self.ln_1.name="ln1"
        self.ln_1.opname="layernorm"
        self.ln_2.name="ln2"
        self.ln_2.opname="layernorm"
       
    def forward(self,x):
        x=x+self.attention(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.config=config
        h_list=[]
        for i in range(config.nlayer):
            h_list.append(TransformerBlock(config))
            h_list[i].ln_1.name+=str(i)
            h_list[i].ln_2.name+=str(i)
            h_list[i].attention.dropout.name+=str(i)
            h_list[i].attention.attn.name+=str(i)

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
        self.transformer.ln_f.name="ln_final"
        self.transformer.ln_f.opname="layernorm"
        self.transformer.wte.name="wte"
        self.transformer.wte.opname="embedding"
        self.transformer.wpe.name="wpe"
        self.transformer.wpe.opname="embedding"
        self.register_buffer("pos_emb_x", torch.arange(0,self.config.seq_len,dtype=torch.long))
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

    def init_grouped_weight(self,module):
        std = 0.02
        if hasattr(module,"name"):
            if module.name.find("c_fc")!=-1 or module.name.find("c_proj")!=-1:
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self,input,label=None):
        input=input.long()
        pos=self.pos_emb_x
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(input) # token embeddings of shape (B, T, n_embd)
        x = tok_emb+pos_emb
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

class PAttention(nn.Module):
    def __init__(self,config,layer_number:int):
        super().__init__()
        self.attn = ParallelAttention(config,config.n_embd,config.n_head,layer_number=layer_number,attention_mask=config.need_attention_mask,dropout=config.attention_dropout,bias=True)
        self.attn.name="attn"
        self.attn.opname="mha"
        self.dropout=nn.Dropout(config.hidden_dropout)
        self.dropout.name="attn_dropout_post"
        self.dropout.opname="dropout"
        # self.register_buffer("attention_mask",nn.Transformer.generate_square_subsequent_mask(config.seq_len))

    def forward(self,x):
        x=self.attn(x)
        x=self.dropout(x)
        return x


class ParallelTransformerBlock(nn.Module):
    def __init__(self,config:TransformerConfig,layer_number:int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention=PAttention(config,layer_number)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = ParallelMLP(config)
        self.mlp.c_fc.name+="ffw"
        self.mlp.c_proj.name+="ffw"
        self.ln_1.name="ln1"
        self.ln_1.opname="layernorm"
        self.ln_2.name="ln2"
        self.ln_2.opname="layernorm"
       
    def forward(self,x):
        x=x+self.attention(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x
    

class ParallelTransformer(nn.Module):
    def __init__(self,config:TransformerConfig):
        super().__init__()
        self.config=config
        h_list=[]
        for i in range(config.nlayer):
            h_list.append(ParallelTransformerBlock(config,i))
            h_list[i].ln_1.name+=str(i)
            h_list[i].ln_2.name+=str(i)
            h_list[i].attention.dropout.name+=str(i)
            h_list[i].attention.attn.name+=str(i)

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
        self.transformer.ln_f.name="ln_final"
        self.transformer.ln_f.opname="layernorm"
        self.transformer.wte.name="wte"
        self.transformer.wte.opname="embedding"
        self.transformer.wpe.name="wpe"
        self.transformer.wpe.opname="embedding"
        self.register_buffer("pos_emb_x", torch.arange(0,self.config.seq_len,dtype=torch.long))
        self.apply(self._init_weights)

    def _init_weights(self,module):
            std = self.config.init_method_std
            with torch.no_grad():
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.MultiheadAttention):
                    torch.nn.init.normal_(module.in_proj_weight,mean=0,std=std)
                    torch.nn.init.normal_(module.out_proj.weight,mean=0,std=std)

                elif isinstance(module, nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def init_grouped_weight(self,module):
        std = 0.02
        if hasattr(module,"name"):
            if module.name.find("c_fc")!=-1 or module.name.find("c_proj")!=-1:
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self,input,label=None):
        input=input.long()
        pos=self.pos_emb_x
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(input) # token embeddings of shape (T, B, n_embd)
        pos_emb=pos_emb.unsqueeze(1)
        x = tok_emb+pos_emb
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