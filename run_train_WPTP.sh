#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=2,3,4

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

export GPUS_NUM=$GPUS_PER_NODE

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)


torchrun ${DISTRIBUTED_ARGS[@]} gpt_train_WPtp.py