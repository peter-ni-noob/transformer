import torch
import torch.distributed
import torch.nn as nn
from torch.nn import functional as F

from typing import Any, Callable, List, Optional, Tuple
from dutil import device_init, divide, get_global_memory_buffer, init_tp_env
from hpConfig import TransformerConfig
import gobalVar
from enum import Enum
import math


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
                        dtype=torch.float32,
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
    Megatron ParallelAttention
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
        self.batch_first=batch_first
        self.layer_number=max(1,layer_number)
        self.config=config
        kv_channels=embed_dim/num_heads
        query_projection_size = kv_channels * num_heads
        num_attention_head = num_heads
        hidden_size = embed_dim
        self.hidden_size=hidden_size
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
           self.mixed_x_layer=ColumnParallelLinear(config,hidden_size, int(query_projection_size + 2 * kv_projection_size),bias,False)
        else:
            assert False,"not implemented!"
        
        self.outmha=RowParallelLinear(config,int(query_projection_size),hidden_size,bias,True)
        if attention_mask:
            self.register_buffer("attention_mask",nn.Transformer.generate_square_subsequent_mask(config.seq_len,dtype=torch.float32))
        else:
            self.register_buffer("attention_mask",None)

        self.num_query_groups_per_partition=int(num_query_groups_per_partition)
        self.num_attention_heads_per_partition=int(num_attention_heads_per_partition)
        self.hidden_size_per_attention_head=int(hidden_size_per_attention_head)
        self.projection_size = int(kv_channels * num_attention_head)
        self.hidden_size_per_partition = divide(self.projection_size, world_size)
        norm_factor=math.sqrt(self.hidden_size_per_attention_head)
        coeff=-1
        if config.apply_query_key_layer_scaling:
            coeff=layer_number
            norm_factor*=layer_number
        self.norm_factor=torch.tensor(coeff,dtype=torch.float32)
        self.coeff=torch.tensor(coeff,dtype=torch.float32)




    def forward(self,hidden_states:torch.Tensor):
        is_batched= hidden_states.ndim ==3
        if is_batched and self.batch_first:
            assert False,"not impl"


        mixed_x_layer=self.mixed_x_layer(hidden_states)

        new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_query_groups_per_partition,
                (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
                ),
            )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query_layer,
            key_layer,
            value_layer) = torch.split(
                mixed_x_layer,
                [
                    (
                        self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                        * self.hidden_size_per_attention_head
                    ),
                    self.hidden_size_per_attention_head,
                    self.hidden_size_per_attention_head
                ],
                dim=3)
         # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
        query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)

        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key_layer = key_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim = 2
            )
            value_layer = value_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim = 2
            )

        #core attenion
        apply_query_key_layer_scaling=self.config.apply_query_key_layer_scaling
        sequence_parallel=self.config.sequence_parallel

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(output_size[2],
                                          output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)
        
        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = get_global_memory_buffer().get_tensor(
            (output_size[0]*output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")
        
        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))
        
        if self.coeff!= -1:
            matmul_result = matmul_result * self.coeff
        
        if self.attention_mask is not None:
            attn_mask = self.attention_mask

            # [sq, sk] -> [1, sq, sk]
            attn_mask = attn_mask.unsqueeze(0)
            matmul_result = matmul_result + attn_mask

        attention_probs = F.softmax(matmul_result, dim=-1)
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)
        
         # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        out=self.outmha(context_layer)
        return out


        

def test_PMHA():
    import atexit

    atexit.register(torch.distributed.destroy_process_group)

    device=device_init()
    init_tp_env()

    config=TransformerConfig()

    layer = ParallelAttention(config,embed_dim=768, num_heads=12,attention_type=AttnType.self_attn,layer_number=1,attention_mask=True,dropout=0.0,bias=True,batch_first=False).to(device)
    input_tensor = torch.randn(1024,2, 768,requires_grad=True,device=device)  # 批次大小为2，输入特征大小为4
    output = layer(input_tensor)
    output.mean().backward()
    print("Output shape:", output.shape)  # 输出形状应为 [2, 4] (8/2)
    print("layer_grad",layer.mixed_x_layer.weight.grad )
    print("layer_grad",layer.outmha.weight.grad )
    print("input_tensor.",input_tensor.grad ) 
        
        




        












if __name__ == "__main__":    # 测试
    test_PMHA()