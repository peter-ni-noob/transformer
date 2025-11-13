import torch
import torch.distributed
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from dataset import GPT2dataset
from dataset import GPT2datasetConfig
from dataset import GPTDistributed_Sampler

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

import torch.utils.data.dataloader as DataLoader
from lr_scheduler import Anneals_Cosine_LR_Scheduler
import time
import os
import sys
from contextlib import nullcontext
import collections
import re
import numpy as np


import gobalVar
from gobalVar import set_global_var

import argparse
from transformerLayer import Transformer
from hpConfig import TransformerConfig, TransformerOptimizerConfig

from dlogger import DLogger
from dutil import get_lrank


def device_init():
    dist_init=int(os.environ.get('RANK', -1)) != -1
    rank=int(os.environ.get('RANK',0))
    if dist_init:
        init_process_group(backend='nccl')
        local_num=torch.cuda.device_count()
        local_rank = rank%local_num
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_var("DEVICE",device)
    return device




# def parse_args():
#     parser = argparse.ArgumentParser(description="GPT Training")
#     parser.add_argument("--tp degree", type=int, help="how many gpus for one tensor parallel group", default=1)
#     args = parser.parse_args()
#     return args


def init_tp_env():
    world_size=1
    if dist.is_available() and dist.is_initialized():
        world_size=dist.get_world_size()
    set_global_var("TPWORLD_SIZE",world_size)
    lrank=get_lrank()
    if lrank=="gpu":
        lrank=0
    else:
        lrank=int(lrank)
    set_global_var("TPRANK",lrank)
    

def main():
    # args=parse_args()
    device=device_init()
    init_tp_env()
    




    dlogger=DLogger()
    # dlogger.info("device:",device,master_only=True)
    modelconfig=TransformerConfig()
    model=Transformer(modelconfig).to(device)
    model.train(True)


    optim_config=TransformerOptimizerConfig()
    optimizer=torch.optim.AdamW(model.parameters(),lr=optim_config.init_lr,weight_decay=optim_config.wd)
    scheduler=Anneals_Cosine_LR_Scheduler()
    gptdataset=GPT2dataset(GPT2datasetConfig())
    sampler=GPTDistributed_Sampler(gptdataset)
    gptdataloader=DataLoader.DataLoader(gptdataset,batch_size=modelconfig.batch_size,shuffle=False,sampler=sampler)


    accumulation_steps=modelconfig.accumulation_steps
    globalBatchSize=modelconfig.batch_size*accumulation_steps

    loss_acc=0.0
    t_begin=time.time()
    real_step=0

    for step,(data,label) in enumerate(gptdataloader):
        l,loss=model(data.to(device),label.to(device))
        loss_acc+=loss.detach().item()
        loss=loss/accumulation_steps
        loss.backward()

        if (step+1)%accumulation_steps==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            loss_acc =loss_acc/accumulation_steps
            lr=scheduler.step_lr()

            for param_group in optimizer.param_groups:
                param_group["lr"]=lr

            optimizer.step()
            optimizer.zero_grad()

            if real_step%modelconfig.log_interval==0:
                t_end=time.time()
                time_per_realstep=t_end - t_begin
                throughput=globalBatchSize*modelconfig.seq_len*modelconfig.log_interval/time_per_realstep
                dlogger.info(f"step:{real_step}, loss:{loss_acc:.6f}, lr:{lr:.6e}, time_cost:{t_end - t_begin:.2f}s, throughout:{int(throughput)}",master_only=True)
                t_begin=time.time()

            #收尾工作
            loss_acc=0.0
            real_step+=1
            if(real_step==4000):
                return




if __name__ == "__main__":
    main()