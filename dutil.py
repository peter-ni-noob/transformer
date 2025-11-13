import os
import torch


def get_lrank():
    dist_init=int(os.environ.get('RANK', -1)) != -1
    rank=int(os.environ.get('RANK',0))
    if dist_init:
        local_num=torch.cuda.device_count()
        local_rank = rank%local_num
        device = local_rank
    else:
        device = "gpu"
    return str(device)

