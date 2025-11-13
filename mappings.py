import torch
import gobalVar


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""


    # Bypass the function if we are using only 1 GPU.
    if gobalVar.TPWORLD_SIZE == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_.contiguous())

    return input_



class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return input_

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _reduce(grad_output), None



class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return grad_output, None


def copy_to_tensor_model_parallel_region(input_):
    """Wrapper for autograd function: forward: copy, backward allreduce"""
    return _CopyToModelParallelRegion.apply(input_)

def reduce_from_tensor_model_parallel_region(input_):
    """Wrapper for autograd function: forward: all reduce, backward copy"""
    return _ReduceFromModelParallelRegion.apply(input_)