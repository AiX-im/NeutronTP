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

# 每个worker切分feature后，将切分后的feature发送给其他worker
def split(local_feature):
    env = DistEnv.env
    src = env.rank
    splits = torch.chunk(local_feature, chunks=env.world_size, dim = 1)
    splits_contiguous = [split.contiguous() for split in splits]
    with env.timer.timing_cuda('broadcast'):
        # 使用 PyTorch 分布式通信库进行广播
        recv_list = [torch.zeros_like(splits_contiguous[src]) for _ in range(env.world_size)]
        env.barrier_all()
        dist.all_gather(recv_list, splits_contiguous[src], group=env.world_group) #worker i聚合其他worker的第i个splits
        recv_tensor = torch.Tensor(torch.cat(recv_list, dim = 0))
    return recv_tensor

# 每个worker收集全部切分feature并将其拼接为完整的feature, 每个worker持有本地节点的完整feature
def gather(local_feature):
    env = DistEnv.env
    src = env.rank
    splits = torch.chunk(local_feature, chunks=env.world_size, dim = 0)
    splits_contiguous = [split.contiguous() for split in splits]
    with env.timer.timing_cuda('broadcast'):
        # 使用 PyTorch 分布式通信库进行广播
        recv_list = [torch.zeros_like(splits_contiguous[src]) for _ in range(env.world_size)]
        env.barrier_all()
        dist.all_gather(recv_list, splits_contiguous[src], group=env.world_group) #worker i聚合其他worker的第i个splits
    return torch.Tensor(torch.cat(recv_list, dim = 1))


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
    def forward(ctx, features, adj_full, layers, tag):
        env = DistEnv.env
        ctx.adj_full = adj_full
        ctx.tag = tag
        ctx.nlayers = layers
        if tag == 0:
            features = split(features) #前向图操作开始前切分tensor
        env.barrier_all()
        z_local = torch.zeros_like(features)
        with env.timer.timing_cuda('spmm'):
            spmm(adj_full, features, z_local)  #图操作 无需广播
        env.barrier_all()
        if tag == layers - 1:
            z_local = gather(z_local)  #前向图操作结束后聚合tensor  
        return z_local
    
    @staticmethod
    def backward(ctx, grad_output):
        env = DistEnv.env
        if ctx.tag == ctx.nlayers - 1:
            grad_output = split(grad_output)  #反向图操作开始前切分tensor
        ag = torch.zeros_like(grad_output)
        with env.timer.timing_cuda('spmm'):
            spmm(ctx.adj_full, grad_output, ag)  #图操作 无需广播
        if ctx.tag == 0:
            ag = gather(ag) #反向图操作结束后聚合tensor
        return ag, None, None, None


class TensplitGCN(nn.Module):
    def __init__(self, g, env, hidden_dim=16, nlayers=2):
        super().__init__()
        self.g, self.env = g, env
        in_dim, out_dim = g.features.size(1), g.num_classes
        torch.manual_seed(0)

        self.nlayers = nlayers
        self.layers = nn.ParameterList()
        self.layers.append(nn.Parameter(torch.rand(in_dim, hidden_dim).to(env.device)))
        for i in range(1, nlayers-1):
            self.layers.append(nn.Parameter(torch.rand(hidden_dim, hidden_dim).to(env.device)))
        self.layers.append(nn.Parameter(torch.rand(hidden_dim, out_dim).to(env.device)))

        for weight in self.layers:
            nn.init.xavier_uniform_(weight)

    def forward(self, features):
        hidden_features = features
        #NN
        for i, weight in enumerate(self.layers):
            hidden_features = DistNNLayer.apply(hidden_features, weight)
            if i != len(self.layers) - 1:
                hidden_features = F.relu(hidden_features)
        #Graph
        # print(f"hidden_features {hidden_features.size()} hidden_features.size(0): {hidden_features.size(0)}, hidden_features.size(1): {hidden_features.size(1)}")
        src = DistEnv.env.rank
        for i in range(len(self.layers)):
            DistEnv.env.barrier_all()
            hidden_features = DistGraphLayer.apply(hidden_features, self.g.adj_full, self.nlayers, i)
            DistEnv.env.barrier_all()
        return hidden_features