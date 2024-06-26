import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dist_utils import DistEnv
import torch.distributed as dist

try:
    # spmm = lambda A,B,C: C.addmm_(A,B)
    from spmm_cpp import spmm_cusparse_coo, spmm_cusparse_csr
    def spmm(A,B,C): 
        if DistEnv.env.csr_enabled:
            spmm_cusparse_csr(A.crow_indices().int(), A.col_indices().int(), A.values(), A.size(0), A.size(1), \
                B, C, 1.0, 1.0, DistEnv.env.half_enabled)
        else:
            spmm_cusparse_coo(A.indices()[0].int(), A.indices()[1].int(), A.values(), A.size(0), A.size(1), \
                B, C, 1.0, 1.0, DistEnv.env.half_enabled)
except ImportError as e:
    # print('no spmm cpp:', e)
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
            dist.scatter(recv_list[env.rank], src = env.rank, scatter_list=scatter_list) #scatter_list only in srcrank
        else:
            dist.scatter(recv_list[src], src = src)
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
        # recv_tensor = recv_tensor.cuda()
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
        # recv_tensor = recv_tensor.cpu() 
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


class data_prefetcher():
    def __init__(self, adj_chunks):
        self.adj_chunks = adj_chunks
        self.stream = torch.cuda.Stream()
        
        
    def feature_slices_init(self, feature_chunks, output_chunks):
        self.feature_chunks = feature_chunks
        self.output_chunks = output_chunks
        self.i = 0
        self.j = 0
        self.preload()
    
    def preload(self):
        if self.i < len(self.adj_chunks) - 1 and self.j == 0:
            self.next_feature_chunk = self.feature_chunks[self.i]
            self.i = self.i + 1
            
        if self.j < len(self.adj_chunks[0]) - 1:
            self.next_adj_chunk = self.adj_chunks[self.i][self.j]
            self.next_output_chunk = self.output_chunks[self.j]
            self.j = self.j + 1
            
        if self.j == len(self.adj_chunks[0]) - 1 and self.i < len(self.adj_chunks) - 1:
            self.j = 0    
        
        if self.i == len(self.adj_chunks) - 1 and self.j == len(self.adj_chunks[0]) - 1:
            self.next_adj_chunk = None
            self.next_feature_chunk = None
            self.next_output_chunk = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_adj_chunk = self.next_adj_chunk.cuda(non_blocking=True)
            self.next_feature_chunk = self.next_feature_chunk.cuda(non_blocking=True)
            self.next_output_chunk = self.next_output_chunk.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        adj_chunks = self.next_adj_chunk
        feature_chunks = self.next_feature_chunk
        output_chunks = self.next_output_chunk
        if adj_chunks is not None:
            adj_chunks.record_stream(torch.cuda.current_stream())
        if feature_chunks is not None:
            feature_chunks.record_stream(torch.cuda.current_stream())
        if output_chunks is not None:
            output_chunks.record_stream(torch.cuda.current_stream())
        self.preload()
        return adj_chunks, feature_chunks, output_chunks, self.j - 1


class DistPipeChunkLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, prefetcher, chunk_num, layers, tag):
        env = DistEnv.env
        ctx.prefetcher = prefetcher
        ctx.tag = tag
        ctx.nlayers = layers
        ctx.chunk_num = chunk_num
        if tag == 0:
            features = split(features) #前向图操作开始前切分tensor
        feature_chunk = torch.chunk(features, chunk_num, dim = 0)
        for i in range(len(feature_chunk)):
            feature_chunk[i] = feature_chunk[i].pin_memory()
        Graph_output = [torch.zeros_like(chunk,device="cpu") for chunk in feature_chunk]
        for i in range(len(Graph_output)):
            Graph_output[i] = Graph_output[i].pin_memory()
        prefetcher.feature_slices_init(feature_chunk, Graph_output)
        adj_chunk_dev, feature_chunk_dev, Graph_output_dev, j = prefetcher.next()
        with env.timer.timing('spmm'):
            while adj_chunk_dev is not None:
                spmm(adj_chunk_dev, feature_chunk_dev, Graph_output_dev)  #图操作 无需广播
                Graph_output_tmp = Graph_output_dev.cpu()
                Graph_output[j] = Graph_output[j] + Graph_output_tmp
                adj_chunk_dev, feature_chunk_dev, Graph_output_dev = prefetcher.next()      
        z_local = torch.Tensor(torch.cat(Graph_output, dim = 0))
        if tag == layers - 1:
            z_local = gather(z_local)  #前向图操作结束后聚合tensor  
        
        del Graph_output, feature_chunk
        return z_local
    
    @staticmethod
    def backward(ctx, grad_output):
        env = DistEnv.env
        if ctx.tag == ctx.nlayers - 1:
            grad_output = split(grad_output)  #反向图操作开始前切分tensor
        feature_chunk = torch.chunk(grad_output, ctx.chunk_num, dim = 0)
        for i in range(len(feature_chunk)):
            feature_chunk[i] = feature_chunk[i].pin_memory()
        Graph_output = [torch.zeros_like(chunk,device="cpu") for chunk in feature_chunk]
        for i in range(len(Graph_output)):
            Graph_output[i] = Graph_output[i].pin_memory()
        ctx.prefetcher.feature_slices_init(feature_chunk, Graph_output)
        adj_chunk_dev, feature_chunk_dev, Graph_output_dev, j = ctx.prefetcher.next()
        with env.timer.timing('spmm'):
            while adj_chunk_dev is not None:
                spmm(adj_chunk_dev, feature_chunk_dev, Graph_output_dev)  #图操作 无需广播
                Graph_output_tmp = Graph_output_dev.cpu()
                Graph_output[j] = Graph_output[j] + Graph_output_tmp
                adj_chunk_dev, feature_chunk_dev, Graph_output_dev = ctx.prefetcher.next()      
        ag = torch.Tensor(torch.cat(Graph_output, dim = 0))
        if ctx.tag == 0:
            ag = gather(ag) #反向图操作结束后聚合tensor
            
        del Graph_output, feature_chunk
        return ag, None, None, None, None



class DistChunkLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, adj_chunk,chunk_num, layers, tag):
        env = DistEnv.env
        ctx.adj_chunk = adj_chunk
        ctx.tag = tag
        ctx.nlayers = layers
        ctx.chunk_num = chunk_num
        if tag == 0:
            features = split(features) #前向图操作开始前切分tensor
        feature_chunk = torch.chunk(features, chunk_num, dim = 0)
        Graph_output = [torch.zeros_like(chunk,device="cpu") for chunk in feature_chunk]
        with env.timer.timing('spmm'):
            for i in range(len(adj_chunk)):
                feature_chunk_dev = feature_chunk[i].cuda()
                for j in range(len(adj_chunk[0])):
                    adj_chunk_dev = adj_chunk[i][j].cuda()
                    Graph_output_dev = Graph_output[j].cuda()
                    spmm(adj_chunk_dev, feature_chunk_dev, Graph_output_dev)  #图操作 无需广播
                    Graph_output_tmp = Graph_output_dev.cpu()
                    Graph_output[j] = Graph_output[j] + Graph_output_tmp      
        z_local = torch.Tensor(torch.cat(Graph_output, dim = 0))
        if tag == layers - 1:
            z_local = gather(z_local)  #前向图操作结束后聚合tensor  
        return z_local
    
    @staticmethod
    def backward(ctx, grad_output):
        env = DistEnv.env
        if ctx.tag == ctx.nlayers - 1:
            grad_output = split(grad_output)  #反向图操作开始前切分tensor
        feature_chunk = torch.chunk(grad_output, ctx.chunk_num, dim = 0)
        Graph_output = [torch.zeros_like(chunk,device="cpu") for chunk in feature_chunk]
        with env.timer.timing('spmm'):
            for i in range(len(ctx.adj_chunk)):
                feature_chunk_dev = feature_chunk[i].cuda()
                for j in range(len(ctx.adj_chunk[0])):
                    adj_chunk_dev = ctx.adj_chunk[i][j].cuda()
                    Graph_output_dev = Graph_output[j].cuda()
                    spmm(adj_chunk_dev, feature_chunk_dev, Graph_output_dev)  #图操作 无需广播
                    Graph_output_tmp = Graph_output_dev.cpu()
                    Graph_output[j] = Graph_output[j] + Graph_output_tmp
        ag = torch.Tensor(torch.cat(Graph_output, dim = 0))
        if ctx.tag == 0:
            ag = gather(ag) #反向图操作结束后聚合tensor
        return ag, None, None, None, None


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
        device = torch.device('cpu')
        self.layers.append(nn.Parameter(torch.rand(in_dim, hidden_dim).to(device)))
        for i in range(1, nlayers-1):
            self.layers.append(nn.Parameter(torch.rand(hidden_dim, hidden_dim).to(device)))
        self.layers.append(nn.Parameter(torch.rand(hidden_dim, out_dim).to(device)))

        for weight in self.layers:
            nn.init.xavier_uniform_(weight)
            
        self.prefetcher = data_prefetcher(self.g.adj_chunks)

    def forward(self, features):
        hidden_features = features
        #NN
        for i, weight in enumerate(self.layers):
            hidden_features = DistNNLayer.apply(hidden_features, weight)
            if i != len(self.layers) - 1:
                hidden_features = F.relu(hidden_features)

        #Graph
        # print(f"hidden_features {hidden_features.size()} hidden_features.size(0): {hidden_features.size(0)}, hidden_features.size(1): {hidden_features.size(1)}")
        # hidden_features = torch.ones((hidden_features.shape[0], 41)).to(DistEnv.env.device)
        env = DistEnv.env
        device = torch.device('cpu')
        dim_diff = env.world_size - hidden_features.shape[1] % env.world_size
        if dim_diff > 0:
            padding_tensor = torch.zeros((hidden_features.size(0), dim_diff), dtype=hidden_features.dtype, device=device)
            hidden_features = torch.cat((hidden_features, padding_tensor), dim=1)
        
        # src = env.rank
        # for i in range(len(self.layers)):
        #     hidden_features = DistPipeChunkLayer.apply(hidden_features, self.prefetcher, self.g.chunk_num, self.nlayers, i)
        # if dim_diff > 0:
        #     hidden_features = hidden_features[:, : -dim_diff].contiguous()
        # return hidden_features
    
        for i in range(len(self.layers)):
            hidden_features = DistChunkLayer.apply(hidden_features, self.g.adj_chunks,self.g.chunk_num, self.nlayers, i)
        if dim_diff > 0:
            hidden_features = hidden_features[:, : -dim_diff].contiguous()
        return hidden_features
        # for i in range(len(self.layers)):
        #     hidden_features = DistGraphLayer.apply(hidden_features, self.g.adj_full, self.nlayers, i)
        # if dim_diff > 0:
        #     hidden_features = hidden_features[:, : -dim_diff].contiguous()
        # return hidden_features
        # for i in range(len(self.layers)):
        #     hidden_features = DistGraphLayer.apply(hidden_features, self.g.adj_parts, self.nlayers, i)
        # if dim_diff > 0:
        #     hidden_features = hidden_features[:, : -dim_diff].contiguous()
        # return hidden_features