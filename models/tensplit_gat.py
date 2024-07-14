import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from dist_utils import DistEnv

try:
    from spmm_cpp import spmm_cusparse_coo, spmm_cusparse_csr
    def spmm(A, B, C): 
        if DistEnv.env.csr_enabled:
            spmm_cusparse_csr(A.crow_indices().int(), A.col_indices().int(), A.values(), A.size(0), A.size(1), 
                              B, C, 1.0, 1.0, DistEnv.env.half_enabled)
        else:
            spmm_cusparse_coo(A.indices()[0].int(), A.indices()[1].int(), A.values(), A.size(0), A.size(1), 
                              B, C, 1.0, 1.0, DistEnv.env.half_enabled)
except ImportError as e:
    print('no spmm cpp:', e)
    spmm = lambda A, B, C: C.addmm_(A, B)


def split(local_feature):
    env = DistEnv.env
    splits = torch.chunk(local_feature, chunks=env.world_size, dim=1)
    splits_contiguous = [split.contiguous() for split in splits]
    recv_list = [torch.zeros_like(splits_contiguous[env.rank]) for _ in range(env.world_size)]
    dist.all_to_all(recv_list, splits_contiguous, group=env.world_group)
    return torch.cat(recv_list, dim=0)


def gather(local_feature):
    env = DistEnv.env
    splits = torch.chunk(local_feature, chunks=env.world_size, dim=0)
    splits_contiguous = [split.contiguous() for split in splits]
    recv_list = [torch.zeros_like(splits_contiguous[env.rank]) for _ in range(env.world_size)]
    dist.all_to_all(recv_list, splits_contiguous, group=env.world_group)
    return torch.cat(recv_list, dim=1)


def compute_attention(features, adj_indices, attention_weight, device):
    edge_features = torch.cat((features[adj_indices[0, :], :], features[adj_indices[1, :], :]), dim=1)
    att_input = F.leaky_relu(torch.mm(edge_features, attention_weight).squeeze())
    att_input = torch.sparse_coo_tensor(adj_indices, att_input, (features.size(0), features.size(0))).to(device)
    attention = torch.sparse.softmax(att_input, dim=1)
    return attention


class DistNNLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight):
        ctx.save_for_backward(features, weight)
        z_local = features
        with DistEnv.env.timer.timing_cuda('mm'):
            z_local = torch.mm(z_local, weight)
        return z_local

    @staticmethod
    def backward(ctx, grad_output):
        features, weight = ctx.saved_tensors
        with DistEnv.env.timer.timing_cuda('mm'):
            grad_features = torch.mm(grad_output, weight.t())
            grad_weight = torch.mm(features.t(), grad_output)
        with DistEnv.env.timer.timing_cuda('all_reduce'):
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
            features = split(features)
        z_local = torch.zeros_like(features)
        with env.timer.timing_cuda('spmm'):
            spmm(adj_full, features, z_local)
        if tag == layers - 1:
            z_local = gather(z_local)
        return z_local

    @staticmethod
    def backward(ctx, grad_output):
        env = DistEnv.env
        if ctx.tag == ctx.nlayers - 1:
            grad_output = split(grad_output)
        ag = torch.zeros_like(grad_output)
        with env.timer.timing_cuda('spmm'):
            spmm(ctx.adj_full, grad_output, ag)
        if ctx.tag == 0:
            ag = gather(ag)
        return ag, None


class TensplitGAT(nn.Module):
    def __init__(self, g, env, hidden_dim=16, nlayers=2):
        super().__init__()
        self.g, self.env = g, env
        in_dim, out_dim = g.features.size(1), g.num_classes
        torch.manual_seed(0)

        self.nlayers = nlayers
        self.weights = nn.ParameterList([nn.Parameter(torch.rand(in_dim if i == 0 else hidden_dim, hidden_dim).to(env.device)) for i in range(nlayers)])
        self.weights.append(nn.Parameter(torch.rand(hidden_dim, out_dim).to(env.device)))
        self.attention_weights = nn.ParameterList([nn.Parameter(torch.rand(2 * hidden_dim, 1).to(env.device)) for _ in range(nlayers)])
        self.attention_weights.append(nn.Parameter(torch.rand(out_dim * 2, 1).to(env.device)))

        self.hidden_features = None

    def forward(self, features):
        hidden_features = features
        device = self.env.device
        adj_indices = self.g.adj._indices()
        
        # 检查 self.hidden_features 是否存在
        if self.hidden_features is None:
            # 如果没有历史hidden_features，则默认attention为全1的稀疏张量
            attention = torch.sparse_coo_tensor(adj_indices, torch.ones(self.g.adj._nnz()).to(device), self.g.adj.size()).to(device)
        else:
            # 使用保存在CPU的self.hidden_features来计算边注意力系数
            local_hidden_features = self.hidden_features.to(device)
            attention = compute_attention(local_hidden_features, adj_indices, self.attention_weights[0], device)
        
        # 遍历每一层
        for l in range(len(self.weights)):
            all_Hw = DistNNLayer.apply(hidden_features, self.weights[l])
            hidden_features = DistGraphLayer.apply(all_Hw, attention, self.nlayers, l)
            hidden_features = F.elu(hidden_features)
        
        # 将当前hidden_features存储到CPU，作为历史hidden_features
        self.hidden_features = hidden_features.to('cpu')

        # all_Hw = DistNNLayer.apply(hidden_features, self.weights[-1])
        # edge_features = torch.cat((all_Hw[adj_indices[0, :], :], all_Hw[adj_indices[1, :]]), dim=1)
        # att_input = F.leaky_relu(torch.mm(edge_features, self.attention_weights[-1]).squeeze())
        # attention = torch.sparse_coo_tensor(adj_indices, att_input, self.g.adj.size()).to(device)
        # attention = torch.sparse.softmax(attention, dim=1)

        # outputs = DistGraphLayer.apply(all_Hw, attention, self.nlayers, len(self.weights) - 1)

        # 返回log softmax的输出
        return F.log_softmax(hidden_features, 1)
