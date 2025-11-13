import torch
import torch.distributed
import torch.nn as nn
from torch.nn import functional as F

from typing import Any, Callable, List, Optional, Tuple
from dutil import divide
from hpConfig import ModelParallelConfig
import gobalVar



from torch.nn.parameter import Parameter




class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    """

    def __init__(
        self,
        input_size,
        output_size,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add
        self.config = config

        world_size = gobalVar.TPWORLD_SIZE
        rank = gobalVar.TPRANK

        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:
            if config.use_cpu_initialization:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition, self.input_size, dtype=config.params_dtype
                    )
                )
                if config.perform_initialization:
                    self.master_weight = _initialize_affine_weight_cpu(
                        self.weight,
                        self.output_size,
                        self.input_size,
                        self.output_size_per_partition,
                        0,
                        init_method,
                        stride=stride,
                        return_master_weight=keep_master_weight_for_test,
                        rank=rank,
                        world_size=world_size,
                    )
            else:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                if config.perform_initialization:
                    _initialize_affine_weight_gpu(
                        self.weight,
                        init_method,
                        partition_dim=0,
                        stride=stride,
                        is_expert=self.is_expert,
                    )

            setattr(self.weight, "allreduce", not (self.is_expert and self.expert_parallel))
        else:
            self.weight = None

        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
            setattr(self.bias, "allreduce", not (self.is_expert and self.expert_parallel))
        else:
            self.register_parameter("bias", None)

        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and world_size <= 1:
            warnings.warn(
                "`sequence_parallel` is set to `True`, but tensor model parallel size "
                f"is {world_size}. Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.allreduce_dgrad = (
            world_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce
        )

        if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:
            raise RuntimeError(
                "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                "module is not found. To use gradient_accumulation_fusion you must "
                "install APEX with --cpp_ext and --cuda_ext. For example: "
                'pip install --global-option="--cpp_ext" --global-option="--cuda_ext ." '
                "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                "gradient accumulation fusion."
            )
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

        if self.allreduce_dgrad and self.sequence_parallel:
            raise RuntimeError(
                "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
            )

        # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f"{prefix}_extra_state"
            )
        )

    def _forward_impl(self, input, weight, *args, **kwargs):
        if not weight.requires_grad:
            return linear_with_frozen_weight(input, weight, *args, **kwargs)
        else:
            return linear_with_grad_accumulation_and_async_allreduce(input, weight, *args, **kwargs)

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        """Forward of ColumnParallelLinear

        Args:
            input_:
                3D tensor whose order of dimension is [sequence, batch, hidden]
            weight (optional):
                weight tensor to use, compulsory when skip_weight_param_allocation is True.
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `gather_output` arg in the constructor will be used.

        Returns:
            - output
            - bias

        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        bias = self.bias if not self.skip_bias_add else None

        if (
            self.allreduce_dgrad
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

        if self.config.defer_embedding_wgrad_compute:
            if (
                self.config.wgrad_deferral_limit == 0
                or len(self.embedding_activation_buffer) < self.config.wgrad_deferral_limit
            ):
                self.embedding_activation_buffer.append(input_parallel)

        # Matrix multiply.
        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context is True:
                if not HAVE_TE:
                    assert (
                        self.config.cpu_offloading is False
                    ), "CPU Offloading cannot be enabled while TE is not present"
                else:
                    input_parallel.activation_offloading = self.config.cpu_offloading_activations

        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
            grad_output_buffer=(
                self.grad_output_buffer if self.config.defer_embedding_wgrad_compute else None
            ),
            wgrad_deferral_limit=(
                self.config.wgrad_deferral_limit
                if self.config.defer_embedding_wgrad_compute
                else None
            ),
            tp_group=self.tp_group,
        )

        gather_output = self.gather_output
        # Use the runtime gather output if it's set explicitly.
        if runtime_gather_output is not None:
            gather_output = runtime_gather_output

        if gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias