import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dist_utils import DistEnv
import torch.distributed as dist

try:
    from spmm_cpp import spmm_cusparse_coo, spmm_cusparse_csr
    def spmm(A,B,C): 
        if DistEnv.env.csr_enabled:
            spmm_cusparse_csr(A.crow_indices().int(), A.col_indices().int(), A.values(), A.size(0), A.size(1), \
                B, C, 1.0, 1.0, DistEnv.env.half_enabled)
        else:
            spmm_cusparse_coo(A.indices()[0].int(), A.indices()[1].int(), A.values(), A.size(0), A.size(1), \
                B, C, 1.0, 1.0, DistEnv.env.half_enabled)
except ImportError as e:
    print('no spmm cpp:', e)
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
        dist.all_to_all(recv_list, splits_contiguous, group=env.world_group) #worker i聚合其他worker的第i个splits
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
        dist.all_to_all(recv_list, splits_contiguous, group=env.world_group) #worker i聚合其他worker的第i个splits
    return torch.Tensor(torch.cat(recv_list, dim = 1))


# def compute_attention(features, adj_indices, attention_weight, device):
#     edge_features = torch.cat((features[adj_indices[0, :], :], features[adj_indices[1, :], :]), dim=1)
#     att_input = F.leaky_relu(torch.mm(edge_features, attention_weight).squeeze())
#     att_input = torch.sparse_coo_tensor(adj_indices, att_input, (features.size(0), features.size(0))).to(device)
#     attention = torch.sparse.softmax(att_input, dim=1)
#     return attention

def compute_attention(features_src, features_dst, attention_weight):
    edge_features = torch.cat((features_src, features_dst), dim=1)
    att_input = F.leaky_relu(torch.mm(edge_features, attention_weight).squeeze())
    return att_input

def get_attention(local_feature, adj_full, split_size, attention_weight):
    env = DistEnv.env
    device = env.device
    feature_bcast = torch.zeros_like(local_feature)
    
    # 获取稀疏张量的索引和值
    adj_indices = adj_full._indices()
    # adj_values = adj_full._values()
    
    # 初始化attention_local，用于存储相关子图的边注意力系数
    num_local_nodes = local_feature.size(0)
    num_dst_nodes = adj_full.size(1)
    attention_local_indices = []
    attention_local_values = []
    
    # 提取本地子图的索引
    local_mask = (adj_indices[0] >= split_size * env.rank) & (adj_indices[0] < split_size * (env.rank + 1))
    local_adj_indices = adj_indices[:, local_mask]
    
    # 将源节点的索引转换为本地索引
    local_adj_indices[0] -= split_size * env.rank
    
    for worker in range(env.world_size):
        if worker == env.rank:
            feature_bcast = local_feature.clone()
        with env.timer.timing_cuda('broadcast'):
            dist.broadcast(feature_bcast, src=worker)
        
        # 计算边的注意力系数
        dst_start = split_size * worker
        dst_end = split_size * (worker + 1)
        
        # 提取有边的src和dst的特征
        mask = (local_adj_indices[1] >= dst_start) & (local_adj_indices[1] < dst_end)
        src_indices = local_adj_indices[0][mask]
        dst_indices = local_adj_indices[1][mask]
        features_src = local_feature[src_indices]
        features_dst = feature_bcast[dst_indices - dst_start]

        # 计算attention
        att_input = compute_attention(features_src, features_dst, attention_weight)
        
        # 存储计算得到的注意力系数到attention_local
        attention_indices = torch.stack([src_indices, dst_indices, att_input])
        attention_local_indices.append(attention_indices)

    attention_local_indices = torch.cat(attention_local_indices, dim=1).to(device)
    
    # Gather all attention indices from all processes
    attention_global_indices_list = [torch.zeros_like(attention_local_indices) for _ in range(env.world_size)]
    dist.all_gather(attention_global_indices_list, attention_local_indices, group=env.world_group)
    
    attention_global_indices = torch.cat(attention_global_indices_list, dim=1)
    
    attention_shape = (adj_full.size(0), adj_full.size(1))
    attention_values = attention_global_indices[2, :]
    attention_global_indices = attention_global_indices[:2, :]

    attention_global = torch.sparse_coo_tensor(attention_global_indices, attention_values, attention_shape)

    attention_global = torch.sparse.softmax(attention_global, dim=1).coalesce()

    return attention_global


class DistNNLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight):
        ctx.save_for_backward(features, weight)
        z_local = features
        # print(f'rank {DistEnv.env.rank} features.shape {features.shape}')
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
            # print(f'rank{env.rank} before_split {features.shape}')
            features = split(features) #前向图操作开始前切分tensor
            # print(f'rank{env.rank} after_split {features.shape}')
        z_local = torch.zeros_like(features)
        with env.timer.timing_cuda('spmm'):
            spmm(adj_full, features, z_local)  #图操作 无需广播
        if tag == layers - 1:
            # print(f'rank{env.rank} before_gather {z_local.shape}')
            z_local = gather(z_local)  #前向图操作结束后聚合tensor  
            # print(f'rank{env.rank} after_gather {z_local.shape}')
        return z_local
    
    @staticmethod
    def backward(ctx, grad_output):
        env = DistEnv.env
        if ctx.tag == ctx.nlayers - 1:
            # print(f'rank{env.rank} grad_before_split {grad_output.shape}')
            grad_output = split(grad_output)  #反向图操作开始前切分tensor
            # print(f'rank{env.rank} grad_after_split {grad_output.shape}')
        ag = torch.zeros_like(grad_output)
        with env.timer.timing_cuda('spmm'):
            spmm(ctx.adj_full, grad_output, ag)  #图操作 无需广播
        if ctx.tag == 0:
            # print(f'rank{env.rank} grad_before_gather {grad_output.shape}')
            ag = gather(ag) #反向图操作结束后聚合tensor
            # print(f'rank{env.rank} grad_after_gather {grad_output.shape}')
        return ag, None, None, None


class TensplitGAT(nn.Module):
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
        
        self.attention_weights = nn.ParameterList()
        self.attention_weights.append(nn.Parameter(torch.rand((out_dim+1) * 2, 1).to(env.device)))
        
        for weight in self.layers:
            nn.init.xavier_uniform_(weight)

    def forward(self, features):
        hidden_features = features
        #NN
        for i, weight in enumerate(self.layers):
            hidden_features = DistNNLayer.apply(hidden_features, weight)
            if i != len(self.layers) - 1:
                hidden_features = F.relu(hidden_features)
        
                
        env = DistEnv.env
        dim_diff = env.world_size - hidden_features.shape[1] % env.world_size
        if dim_diff > 0:
            # print('dim_diff', dim_diff)
            padding_tensor = torch.zeros((hidden_features.size(0), dim_diff), dtype=hidden_features.dtype, device=env.device)
            # print(f'rank{env.rank} before_pad {hidden_features.shape}')
            hidden_features = torch.cat((hidden_features, padding_tensor), dim=1)
            # print(f'rank{env.rank} after_pad {hidden_features.shape}', torch.sum(hidden_features[:,-1]))
        src = env.rank
        
        # print(self.g.adj_full)
        
        attention = get_attention(hidden_features, self.g.adj_full, self.g.split_size, self.attention_weights[0])
        
        for i in range(len(self.layers)):
            hidden_features = DistGraphLayer.apply(hidden_features, attention, self.nlayers, i)
            # hidden_features = DistGraphLayer.apply(hidden_features, self.g.adj_full, self.nlayers, i)
        if dim_diff > 0:
            hidden_features = hidden_features[:, : -dim_diff].contiguous()
        # print(f'rank{env.rank} output {hidden_features.shape}')
        return hidden_features