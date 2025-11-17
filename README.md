1、docker容器间限速https://www.notion.so/WeightPassing-TP-2a78089d87b580c6a921c39a6f615b92?source=copy_link
2、hpConfig.py中调整batchsize,accumlation step,seq_len等transformer的超参与训练参数来控制模型参数量和激活量
3、建议从同一宿主机的文件夹中启动容器，如两容器通信，一容器使用0文件夹，另一容器使用1文件夹,需要确定MASTER_ADDR（从容器0中ip addr获取eth0的ip，其他容器配置ip均为master node的ip），零号容器使用NODE_RANK=0，1号容器设置NODE_RANK=1
4、容器使用同一个docker虚拟网桥
6、实验目的为比较original TP与WPTP的avg吞吐量
7、数据集在dataset.py中 GPT2datasetConfig修改路径。自回归模型，文本经过tokenization后保存为numpy格式，数据集格式为：（最好有10个文件夹）
-0 -token_float32.npy 形状为(10000,1024)
    -label_float32.npy 形状为(10000,1024)

-1 -token_float32.npy 形状为(10000,1024)
   -label_float32.npy 形状为(10000,1024)

-2 -token_float32.npy 形状为(10000,1024)
   -label_float32.npy 形状为(10000,1024)

