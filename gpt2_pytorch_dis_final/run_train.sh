
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

torchrun \
  --nproc_per_node 8 \
  --nnodes 5 \
  --node_rank 2 \
  --master_addr 192.168.236.11 \
  --master_port 6002 \
  /root/gpt2_pytorch_dis/model_gpt2xl_train_ddp_gacc_dis_nogroupopt_final.py > Ploss40_a64_1112.txt 2>&1


