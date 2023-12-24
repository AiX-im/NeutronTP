import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dist_utils import DistEnv
import torch.distributed as dist

try:
    from spmm_cpp import spmm_cusparse
    # 尝试导入自定义的稀疏矩阵相乘函数，如果失败则使用 PyTorch 默认的矩阵相乘
    spmm = lambda A,B,C: spmm_cusparse(A.indices()[0].int(), A.indices()[1].int(), A.values(), A.size(0), A.size(1), B, C, 1, 1)
except ImportError:
    spmm = lambda A,B,C: C.addmm_(A,B)

# 自定义广播函数，将 local_feature 广播给所有进程
def broadcast(local_adj_parts, local_feature):
    env = DistEnv.env
    z_loc = torch.zeros_like(local_feature)
    feature_bcast = torch.zeros_like(local_feature)
    for src in range(env.world_size):
        if src==env.rank:
            feature_bcast = local_feature.clone()
        # 等待所有进程都准备好再进行广播
        # env.barrier_all()
        with env.timer.timing_cuda('broadcast'):
            # 使用 PyTorch 分布式通信库进行广播
            dist.broadcast(feature_bcast, src=src) #graph
        with env.timer.timing_cuda('spmm'):
            # 调用自定义的稀疏矩阵相乘函数
            spmm(local_adj_parts[src], feature_bcast, z_loc) #nn
    return z_loc

def broadcast_nospmm(local_adj_parts, local_feature):
    env = DistEnv.env
    feature_bcast = torch.zeros_like(local_feature)
    for src in range(env.world_size):
        if src==env.rank:
            feature_bcast = local_feature.clone()
        # 等待所有进程都准备好再进行广播
        # env.barrier_all()
        with env.timer.timing_cuda('broadcast'):
            # 使用 PyTorch 分布式通信库进行广播
            dist.broadcast(feature_bcast, src=src) #graph
    return feature_bcast


class DistNNLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight):
        ctx.save_for_backward(features, weight)
        z_local = features
        # z_local = broadcast(adj_parts, features)
        with DistEnv.env.timer.timing_cuda('mm'):
            z_local = torch.mm(z_local, weight)
        return z_local

    @staticmethod
    def backward(ctx, grad_output):
        features, weight = ctx.saved_tensors
        with DistEnv.env.timer.timing_cuda('mm'):
            # 计算特征的梯度
            grad_features = torch.mm(grad_output, weight.t())
            # 计算权重的梯度
            grad_weight = torch.mm(features.t(), grad_output)
        with DistEnv.env.timer.timing_cuda('all_reduce'):
            # 使用 all_reduce 对梯度进行求和
            DistEnv.env.all_reduce_sum(grad_weight)
        return grad_features, grad_weight

class DistGraphLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, adj_parts):
        ctx.adj_parts = adj_parts
        z_local = broadcast(adj_parts, features) #图操作
        return z_local
    
    @staticmethod
    def backward(ctx, grad_output):
        ag = broadcast(ctx.adj_parts, grad_output)  #只有图操作，所以只需要计算特征的梯度
        return ag, None


class DecoupleGCN(nn.Module):
    def __init__(self, g, env, hidden_dim=16, nlayers=2):
        super().__init__()
        self.g, self.env = g, env
        in_dim, out_dim = g.features.size(1), g.num_classes
        torch.manual_seed(0)

        self.layers = nn.ParameterList()
        self.layers.append(nn.Parameter(torch.rand(in_dim, hidden_dim).to(env.device)))
        for i in range(1, nlayers-1):
            self.layers.append(nn.Parameter(torch.rand(hidden_dim, hidden_dim).to(env.device)))
        self.layers.append(nn.Parameter(torch.rand(hidden_dim, out_dim).to(env.device)))

        for weight in self.layers:
            nn.init.xavier_uniform_(weight)

    def forward(self, features):
        hidden_features = features
        for i, weight in enumerate(self.layers):
            hidden_features = DistNNLayer.apply(hidden_features, weight)
            if i != len(self.layers) - 1:
                hidden_features = F.relu(hidden_features)
        for i in range(len(self.layers)):
            hidden_features = DistGraphLayer.apply(hidden_features, self.g.adj_parts)
        return hidden_features