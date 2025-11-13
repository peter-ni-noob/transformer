import numpy as np
import torch.utils.data.dataset as Dataset
import os
from dataclasses import dataclass
from torch.utils.data.sampler import Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

@dataclass
class GPT2datasetConfig:
    main_path:str="/data/dataset_1b"
    token_num:int=100*10000 #827*10000

@dataclass
class  GPT2EvalDatasetConfig:
    main_path:str="/data/lambada"
    token_num:int=5153


class GPT2dataset(Dataset.Dataset):
    def __init__(self,config:GPT2datasetConfig):
        self.main_path=config.main_path
        self.config=config
        self.inputbuffer=None
        self.labelbuffer=None
        self.preBlock=-1

    def __getitem__(self,index):
        numBlock=index//10000
        numidx=index%10000
        if numBlock!=self.preBlock:
            self.inputbuffer=np.load(os.path.join(self.main_path,str(numBlock),"token_float32.npy"))
            self.labelbuffer=np.load(os.path.join(self.main_path,str(numBlock),"label_float32.npy"))
            self.preBlock=numBlock
        data=self.inputbuffer[numidx]
        label=self.labelbuffer[numidx]
        return data,label
    
    def __len__(self):
        return self.config.token_num


class GPT2EvalDataset(Dataset.Dataset):
    def __init__(self,config:GPT2EvalDatasetConfig):
        self.main_path=config.main_path
        self.config=config
        self.inputbuffer=None
        self.labelbuffer=None
        self.lossmask_buffer=None
        self.preBlock=-1

    def __getitem__(self,index):
        numBlock=index//10000
        numidx=index%10000
        if numBlock!=self.preBlock:
            self.inputbuffer=np.load(os.path.join(self.main_path,str(numBlock),"token_float32.npy"))
            self.labelbuffer=np.load(os.path.join(self.main_path,str(numBlock),"label_float32.npy"))
            self.lossmask_buffer=np.load(os.path.join(self.main_path,str(numBlock),"loss_mask.npy"))
            self.preBlock=numBlock
        data=self.inputbuffer[numidx]
        label=self.labelbuffer[numidx]
        loss_mask=self.lossmask_buffer[numidx]
        return data,label,loss_mask
    
    def __len__(self):
        return self.config.token_num



class GPTDistributed_Sampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        ddp = int(os.environ.get('RANK', -1)) != -1
        if ddp:
            ddp_rank = int(os.environ['RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])
        else :
            ddp_rank=0
            ddp_world_size=1
        self.rank=ddp_rank
        self.world_size=ddp_world_size
        self.len_per_world=len(self.data_source)//self.world_size
        self.begin_idx=self.rank*self.len_per_world
        self.end_idx=self.rank*self.len_per_world+self.len_per_world
        self.sampler_range=range(self.begin_idx,self.end_idx)

    def __iter__(self):
        # return iter(range(len(self.data_source)))
        return iter(self.sampler_range)
    def __len__(self):
        return self.len_per_world
