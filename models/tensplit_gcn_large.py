import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dist_utils import DistEnv
import torch.distributed as dist

try:
    from spmm_cpp import spmm_cusparse, spmm_cusparse
    def spmm(A,B,C): 
        if DistEnv.env.csr_enabled:
            spmm_cusparse(A.crow_indices().int(), A.col_indices().int(), A.values(), A.size(0), A.size(1), \
                B, C, 1.0, 1.0, DistEnv.env.half_enabled)
        else:
            spmm_cusparse(A.indices()[0].int(), A.indices()[1].int(), A.values(), A.size(0), A.size(1), \
                B, C, 1.0, 1.0, DistEnv.env.half_enabled)
except ImportError as e:
    print('no spmm cpp:', e)
    spmm = lambda A,B,C: C.addmm_(A,B)


# N-dim padding for all_gather
def even_all_gather(tensor, env):
    world_size = env.world_size
    device = env.device
    tensor_size = torch.tensor(tensor.size(), device=device)
    all_tensor_sizes = [torch.zeros_like(tensor_size) for _ in range(world_size)]
    dist.all_gather(all_tensor_sizes, tensor_size, group=env.world_group)

    max_tensor_size = max(all_tensor_sizes)
    stacked_tensor = torch.stack(all_tensor_sizes)
    max_tensor_size, _ = torch.max(stacked_tensor, dim=0)

    size_diff = False
    if not torch.equal(tensor_size, max_tensor_size):
        size_diff = True
        pad_tensor = torch.zeros(max_tensor_size.tolist())
        row, col = tensor_size.shape
        pad_tensor[ : row, : col]= tensor
        print(pad_tensor, tensor)
    else:
        pad_tensor = tensor
    
    recv_list = [torch.zeros_like(pad_tensor) for _ in range(env.world_size)]
    dist.all_gather(recv_list, pad_tensor, group=env.world_group) #
    return recv_list

def all_to_all(recv_list,data):
    env = DistEnv.env
    for src in range(env.world_size):
        if env.rank == src:
            scatter_list = data
            dist.scatter(recv_list[env.rank], src = env.rank, scatter_list=scatter_list)
        else:
            dist.scatter(recv_list[env.rank], src = src)
    return recv_list

# def all_to_all(recv_list,data):
#     env = DistEnv.env
#     if env.rank == 1:
#         scatter_list = data
#         dist.scatter(recv_list[0], src = 1, scatter_list=scatter_list)
#     else:
#         dist.scatter(recv_list[0], src = 1)
#     return recv_list

# 每个worker切分feature后，将切分后的feature发送给其他worker
def split(local_feature):
    env = DistEnv.env
    src = env.rank
    splits = torch.chunk(local_feature, chunks=env.world_size, dim = 1)
    splits_contiguous = [split.contiguous() for split in splits]
    with env.timer.timing('broadcast'):
        # 使用 PyTorch 分布式通信库进行广播
        recv_list = [torch.zeros_like(splits_contiguous[src]) for _ in range(env.world_size)]
        all_to_all(recv_list, splits_contiguous) #worker i聚合其他worker的第i个splits
        recv_tensor = torch.Tensor(torch.cat(recv_list, dim = 0))
    return recv_tensor

# 每个worker收集全部切分feature并将其拼接为完整的feature, 每个worker持有本地节点的完整feature
def gather(local_feature):
    env = DistEnv.env
    src = env.rank
    splits = torch.chunk(local_feature, chunks=env.world_size, dim = 0)
    splits_contiguous = [split.contiguous() for split in splits]
    with env.timer.timing('broadcast'):
        # 使用 PyTorch 分布式通信库进行广播
        recv_list = [torch.zeros_like(splits_contiguous[src]) for _ in range(env.world_size)]
        all_to_all(recv_list, splits_contiguous) #worker i聚合其他worker的第i个splits
        recv_tensor = torch.Tensor(torch.cat(recv_list, dim = 1))
    return recv_tensor


class DistNNLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight):
        ctx.save_for_backward(features, weight)
        z_local = features
        with DistEnv.env.timer.timing('mm'):
            z_local = torch.mm(z_local, weight)
        return z_local

    @staticmethod
    def backward(ctx, grad_output):
        features, weight = ctx.saved_tensors
        with DistEnv.env.timer.timing('mm'):
            # 计算特征的梯度
            grad_features = torch.mm(grad_output, weight.t())
            # 计算权重的梯度
            grad_weight = torch.mm(features.t(), grad_output)
        with DistEnv.env.timer.timing('all_reduce'):
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
        z_local = torch.zeros_like(features)
        # print('11111111111111111',adj_full.device,features.device,z_local.device)
        with env.timer.timing('spmm'):
            spmm(adj_full, features, z_local)  #图操作 无需广播
        if tag == layers - 1:
            z_local = gather(z_local)  #前向图操作结束后聚合tensor  
        return z_local
    
    @staticmethod
    def backward(ctx, grad_output):
        env = DistEnv.env
        if ctx.tag == ctx.nlayers - 1:
            grad_output = split(grad_output)  #反向图操作开始前切分tensor
        ag = torch.zeros_like(grad_output)
        with env.timer.timing('spmm'):
            spmm(ctx.adj_full, grad_output, ag)  #图操作 无需广播
        if ctx.tag == 0:
            ag = gather(ag) #反向图操作结束后聚合tensor
        return ag, None, None, None


class TensplitGCNLARGE(nn.Module):
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
        # in_worker_parts_num = 8
        # hidden_features = torch.chunk(features, in_worker_parts_num, dim = 0)
        #初始化GPU中的feature空间，初始化CPU中的锁页内存feature
        #NN
        # for i, weight in enumerate(self.layers):
        #     for chunk in hidden_features:
        #         #上传当前chunk的feature至GPU
        #         chunk = chunk.cuda()
        #         chunk = DistNNLayer.apply(chunk, weight)
        #         if i != len(self.layers) - 1:
        #             chunk = F.relu(chunk)
        #         #传回当前chunk的计算结果至CPU
        # hidden_features = torch.Tensor(torch.cat(chunk, dim = 0))
        # print('22222',features.device)
        hidden_features = features
        #NN
        for i, weight in enumerate(self.layers):
            hidden_features = DistNNLayer.apply(hidden_features, weight)
            if i != len(self.layers) - 1:
                hidden_features = F.relu(hidden_features)
        # print('333333',hidden_features.device)
        
        #Graph
        # print(f"hidden_features {hidden_features.size()} hidden_features.size(0): {hidden_features.size(0)}, hidden_features.size(1): {hidden_features.size(1)}")
        # hidden_features = torch.ones((hidden_features.shape[0], 41)).to(DistEnv.env.device)
        env = DistEnv.env
        dim_diff = env.world_size - hidden_features.shape[1] % env.world_size
        if dim_diff > 0:
            padding_tensor = torch.zeros((hidden_features.size(0), dim_diff), dtype=hidden_features.dtype, device=env.device)
            hidden_features = torch.cat((hidden_features, padding_tensor), dim=1)
        src = env.rank
        for i in range(len(self.layers)):
            hidden_features = DistGraphLayer.apply(hidden_features, self.g.adj_full, self.nlayers, i)
        if dim_diff > 0:
            hidden_features = hidden_features[:, : -dim_diff].contiguous()
        return hidden_features