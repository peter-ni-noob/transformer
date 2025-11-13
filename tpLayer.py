import torch
import torch.distributed
import torch.nn as nn
from torch.nn import functional as F

from typing import Any, Callable, List, Optional, Tuple
from dutil import device_init, divide, init_tp_env
from hpConfig import TransformerConfig
import gobalVar
from enum import Enum


from torch.nn.parameter import Parameter

from mappings import copy_to_tensor_model_parallel_region, reduce_from_tensor_model_parallel_region




class AttnType(Enum):
    self_attn=0,
    cross_attn=1







class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    """

    def __init__(
        self,
        config:TransformerConfig,
        input_size,
        output_size,
        bias=True,
        gather_output=False,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

        world_size = gobalVar.TPWORLD_SIZE
        rank = gobalVar.TPRANK

        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
    
        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                device=torch.cuda.current_device(),
                dtype=torch.float32,
            )
        )
        with torch.no_grad():
            torch.nn.init.normal_(self.weight, mean=0.0, std=config.init_method_std)
        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
            )
            with torch.no_grad():
                torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)



    def forward(
        self,
        input_: torch.Tensor,
    ):
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        if not self.gather_output:
            # All-gather across the partitions.
            output = F.linear(input_parallel, self.weight, self.bias)
        else:
            assert False, "Not Implemented gather output in ColumnParallelLinear"
        
        return output
    

class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X
    along its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    """
    def __init__(
        self,
        config: TransformerConfig,
        input_size: int,
        output_size: int,
        bias: bool,
        input_is_parallel: bool=True,
    ):
        super(RowParallelLinear, self).__init__()
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.config = config
        world_size = gobalVar.TPWORLD_SIZE
        rank = gobalVar.TPRANK

        self.input_size_per_partition = divide(input_size, world_size)

        self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
        )
        with torch.no_grad():
            torch.nn.init.normal_(self.weight, mean=0.0, std=config.init_method_std)
        if bias:
           self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
            )
           with torch.no_grad():
                torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert False, "Not Implemented input is not parallel in RowParallelLinear"

        output_parallel = F.linear(input_parallel, self.weight, None)
        output_=reduce_from_tensor_model_parallel_region(output_parallel)
        if self.bias is not None:
            output_ = output_ + self.bias
        return output_ 





def test_ColumnParallelLinear():
    import atexit

    atexit.register(torch.distributed.destroy_process_group)

    device=device_init()
    init_tp_env()

    config=TransformerConfig()

    layer = ColumnParallelLinear(config, input_size=8, output_size=8, bias=True).to(device)
    input_tensor = torch.randn(20,2, 8,requires_grad=True,device=device)  # 批次大小为2，输入特征大小为4
    output = layer(input_tensor)
    output.mean().backward()
    print("Output shape:", output.shape)  # 输出形状应为 [2, 4] (8/2)
    print("layer_grad",layer.weight.grad )
    print("layer_grad",layer.bias.grad ) 



class ParallelAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self,config:TransformerConfig,
        embed_dim,
        num_heads,
        attention_type:AttnType=AttnType.self_attn,
        layer_number:int =1,
        attention_mask:bool = True,
        dropout: float = 0,
        bias: bool = True,
        batch_first: bool =False):
        super(ParallelAttention,self).__init__()
        self.layer_number=max(1,layer_number)
        self.config=config
        kv_channels=embed_dim/num_heads
        query_projection_size = kv_channels * num_heads
        num_attention_head = num_heads
        hidden_size = embed_dim
        if self.config.group_query_attention:
            kv_projection_size = kv_channels * self.config.num_query_groups
        else:
            kv_projection_size=kv_channels*num_attention_head

        self.world_size=gobalVar.TPWORLD_SIZE
        hidden_size_per_attention_head = divide(query_projection_size, num_attention_head)
        num_attention_heads_per_partition = divide(num_attention_head, self.world_size)

        num_query_groups=self.config.num_query_groups
        world_size=self.world_size

        if self.config.group_query_attention:
            if self.config.num_query_groups % self.world_size !=0 :
                assert False,"Currently the num_query_groups should be a multiple of the tensor parallel size"
            num_query_groups_per_partition = divide(num_query_groups, world_size)
        else:
            num_query_groups_per_partition = num_attention_heads_per_partition
        if attention_type==AttnType.self_attn:
            pass
        else:
            assert attention_type==AttnType.cross_attn
            if self.config.group_query_attention:
                assert False,"Grouped query attention not implemented for cross-attention."
            assert query_projection_size == kv_projection_size

        assert embed_dim % num_heads==0,"embed_dim must be divisible by num_heads"

        if attention_type == AttnType.self_attn:
           self.mixed_x_layer=ColumnParallelLinear(config,hidden_size, query_projection_size + 2 * kv_projection_size,bias,False)
        else:
            assert False,"not implemented!"
        
        self.outmha=RowParallelLinear(config,query_projection_size,hidden_size,bias,True)
        self.register_buffer("attention_mask",nn.Transformer.generate_square_subsequent_mask(config.seq_len))

    def forward(self,hidden_states):
        
        












if __name__ == "__main__":    # 测试ColumnParallelLinear
    pass
 