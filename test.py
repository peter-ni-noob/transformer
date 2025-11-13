import torch
import torch.nn.functional as F

# 假设输入是一个批次的3个样本，每个样本有4个特征
input = torch.randn(3, 4)

# 定义权重和偏置（随机初始化）
weight = torch.randn(5, 4).requires_grad_()
bias = torch.randn(5)

# 使用F.linear()进行线性变换
output = F.linear(input, weight, bias)
y=output.mean( )
y.backward()
print(weight.grad)
print(input.grad)  