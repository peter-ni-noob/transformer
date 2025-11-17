#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=172.18.0.2
MASTER_PORT=6000
# 总容器数
NUM_NODES=2
NODE_RANK=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

export CUDA_VISIBLE_DEVICES=$NODE_RANK


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
)

# gpt_train_origintp.py
# gpt_train_WPtp.py
torchrun ${DISTRIBUTED_ARGS[@]} ./gpt_train_origintp.py