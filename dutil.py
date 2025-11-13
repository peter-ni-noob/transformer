import os
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from gobalVar import set_global_var

def get_lrank():
    dist_init=int(os.environ.get('RANK', -1)) != -1
    rank=int(os.environ.get('RANK',0))
    if dist_init:
        local_rank =int(os.environ['LOCAL_RANK'])
        return local_rank
    return 0

def get_rank():
    dist_init=int(os.environ.get('RANK', -1)) != -1
    rank=int(os.environ.get('RANK',0))
    if dist_init:
        return rank
    return 0


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator



def device_init():
    dist_init=int(os.environ.get('RANK', -1)) != -1
    # rank=int(os.environ.get('RANK',0))
    if dist_init:
        init_process_group(backend='nccl')
        # 分布式启动device_count无法正确获取VISIBLE_DEVICES数量
        # local_num=torch.cuda.device_count()
        # local_num=int(os.environ.get('GPUS_NUM', 0))
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_var("DEVICE",device)
    return device



def init_tp_env():
    world_size=1
    if dist.is_available() and dist.is_initialized():
        world_size=dist.get_world_size()
    set_global_var("TPWORLD_SIZE",world_size)
    lrank=get_lrank()
    set_global_var("TPLOCAL_RANK",lrank)
    rank=get_rank()
    set_global_var("TPRANK",rank)