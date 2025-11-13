import os
import torch


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
