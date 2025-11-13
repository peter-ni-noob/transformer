## 概览 (Big picture)

这是一个以 PyTorch/DistributedDataParallel (DDP) 为主的 GPT-2 风格训练仓库。主要模块划分：

- 数据加载: `dataset.py` 实现了按 block (每 block 包含 10000 条样本) 从磁盘的 numpy 文件按需加载的 `GPT2dataset` 和 `GPT2EvalDataset`，以及基于环境变量 `RANK`/`WORLD_SIZE` 的 `GPTDistributed_Sampler`。
- 模型与训练: 根目录下与 `gpt2_pytorch_dis_final/` 中有多份训练脚本（例如 `model_gpt2xl_train_ddp_gacc_dis_nogroupopt_final.py`），这些文件定义了模型（`GPT`、`TransformerBlock` 等）、训练循环、优化器与 scheduler。
- 工具/脚本: `gpt2_pytorch_dis_final/run_train.sh` 给出了使用 `torchrun` 启动分布式训练的示例（8 GPU × 多节点），并设置了若干环境变量。

设计要点（可用信息）:
- 数据以分块 numpy 文件存储：每个 block 下有 `token_float32.npy`/`label_float32.npy`（eval 还有 `loss_mask.npy`）。dataset 会在跨 block 时按需 load。
- DDP 判定依赖环境变量 `RANK`（存在即认为是 DDP），并通过 `LOCAL_RANK`/`WORLD_SIZE` 设置 device 与进程组（backend='nccl'）。
- 训练采用梯度累积（`accumulation_steps`）与显式 `no_sync` when using DDP，示例代码中用 `model.no_sync` 控制同步。
- 权重保存与加载：state dict 存在 `model_state_dict` 字段；仓库内有 `NOddp`/`save_dic_toCPU` 等 helper 处理 `module.` 前缀与 CPU/GPU 转移。

## 立即可用任务 & 命令示例

- 单节点多卡 (与仓库示例等价):

  ```bash
  # 注意：示例中设置了环境变量以避免 protobuf/连接问题
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

  torchrun --nproc_per_node 8 /path/to/model_gpt2xl_train_ddp_gacc_dis_nogroupopt_final.py
  ```

- 多节点示例见 `gpt2_pytorch_dis_final/run_train.sh`，主要通过 `torchrun --nproc_per_node <gpus> --nnodes <nodes> --node_rank <rank> --master_addr <ip> --master_port <port> ...` 启动。

## 搜索与修改建议（给 AI 代理的具体指令）

- 当需要修改数据处理或排查数据问题，优先打开 `dataset.py`：注意数据分块 (block size=10000) 与文件名（`token_float32.npy`/`label_float32.npy`）。不要假设一次性把全部数据 load 到内存。
- 当修改训练循环（学习率、保存、累积逻辑）时，参阅 `model_gpt2xl_train_ddp_gacc_dis_nogroupopt_final.py`（根目录与 `gpt2_pytorch_dis_final/` 下有多个变体）。常见的变体标签含义：
  - `ddp`：支持分布式训练（env RANK/WORLD_SIZE）
  - `gacc`：梯度累积（gacc = gradient accumulation）
  - `dis`：有 discriminator/额外模块（仓库内有 `net_d.py`）
  - `nogroupopt` / `groupopt`：不同的参数分组/优化器策略

- 查找权重与 checkpoint 代码：搜索 `savePath`/`loadPath`、`torch.save`、`torch.load`、`model_state_dict`、`NOddp`。保存通常在 `ddp_rank==0` 条件下执行。

## 项目约定与注意点（不要违反）

- 不要改动 dataset 的 block size 或 file layout，除非同时更新所有生成数据的 pipeline（模型依赖该格式）。
- 训练脚本通过检查 `RANK` 环境变量来决定是否走 DDP 路径；测试或本地调试可通过不设置这些变量来触发单进程代码路径。
- checkpoint load 有宽松的 `strict=False`（model.load_state_dict(..., strict=False)），修改模型结构时要注意名字匹配和 `NOddp` 的清理逻辑。

## 常见快速定位点（文件示例）

- 数据加载：`dataset.py`（块加载、`GPTDistributed_Sampler`）
- 训练主逻辑（示例/变体）：`model_gpt2xl_train_ddp_gacc_dis_nogroupopt_final.py`（根目录）和 `gpt2_pytorch_dis_final/model_gpt2xl_train_ddp_gacc_dis_nogroupopt_final.py`
- LR scheduler：`lr_scheduler.py`（仓库里的自定义余弦 annealing 实现）
- 训练启动脚本示例：`gpt2_pytorch_dis_final/run_train.sh`
- 小工具/评估：`model_eval_acc_lambada_final.py`、`net_d.py`、`trans_PtoN_weight.py`

## 交付与集成点

- DDP 与 torchrun：按照 `run_train.sh` 的示例设置 `--nproc_per_node`、`--nnodes` 等并保证 `MASTER_ADDR`/`MASTER_PORT` 能互通。
- 数据提供方需生成与 dataset.py 约定一致的 numpy block 文件夹结构：`<main_path>/<block_index>/{token_float32.npy,label_float32.npy}`。

## 如果不确定，AI 代理应先做的三件事

1. 打开 `dataset.py`、要修改的训练脚本和 `run_train.sh` 来确认数据/启动约定。
2. 在修改分布式或保存逻辑前，搜索 `RANK|WORLD_SIZE|LOCAL_RANK|NOddp|model_state_dict|savePath|loadPath`，确保不破坏 checkpoint 兼容性。
3. 在改动后，运行小规模本地（non-DDP）测试来快速验证行为，再扩展到 `torchrun` 多 GPU 运行。

---

如果你希望我把某一段现有训练脚本的片段合并到说明中（例如根目录 vs `gpt2_pytorch_dis_final/` 两个版本的差异），或者把说明翻译为英文，请告诉我要补充的部分或目标语言。
