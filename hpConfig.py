from dataclasses import dataclass


@dataclass
class TransformerConfig:
    n_embd:int=768
    hidden_dropout:float=0.0
    attention_dropout:float=0.0
    batch_size:int=1
    seq_len:int=1024
    nlayer:int=3
    vocab_size:int=50257
    n_head:int=12
    accumulation_steps:int=1
    log_interval:int=10
    init_method_std:float=0.02

    group_query_attention:bool=False
    sequence_parallel:bool=False
    apply_query_key_layer_scaling:bool=False
    num_query_groups:int=1

 
@dataclass
class TransformerOptimizerConfig:
    init_lr=0.0
    wd=1e-1

@dataclass
class ModelParallelConfig:
    tensor_model_parallel_size: int = 1
    perform_initialization: bool = True