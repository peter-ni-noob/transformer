import os
from typing import Callable, Optional
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from contextlib import contextmanager, nullcontext

from gobalVar import set_global_var
import operator

from functools import lru_cache, reduce, wraps


# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

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
    _set_global_memory_buffer()





class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context: Optional[Callable] = None):
        """
        Returns (potentially) a sub-tensor from the self.buffer for the given shape.
        """
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            mem_alloc_context = mem_alloc_context if mem_alloc_context else nullcontext
            with mem_alloc_context():
                self.buffer[(name, dtype)] = torch.empty(
                    required_len,
                    dtype=dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)
    
def _set_global_memory_buffer():
    """Initialize global buffer."""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, "global memory buffer is already initialized"
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()

def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, "global memory buffer is not initialized"
    return _GLOBAL_MEMORY_BUFFER