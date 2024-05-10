import torch
import datetime

# 预处理函数，用于规范化特征并生成邻接矩阵
def preprocess(name, attr_dict, preprocess_for):  # normalize feature and make adj matrix from edge index
    begin = datetime.datetime.now()
    print(name, preprocess_for, 'preprocess begin', begin)
    # 规范化特征，使得每行的和为1
    attr_dict["features"] = attr_dict["features"] / attr_dict["features"].sum(1, keepdim=True).clamp(min=1)
    if preprocess_for == 'GCN':  # make the coo format sym lap matrix
        # 对于GCN，创建对称拉普拉斯矩阵（coo格式）
        attr_dict['adj'] = sym_normalization(attr_dict['edge_index'], attr_dict['num_nodes'])
    elif preprocess_for == 'GAT':
        # 对于GAT，直接使用边的索引作为邻接矩阵
        attr_dict['adj'] = attr_dict['edge_index']
    # 删除原始的边索引，以释放内存
    attr_dict.pop('edge_index')
    print(preprocess_for, 'preprocess done', datetime.datetime.now() - begin)
    return attr_dict

# 添加自环边
def add_self_loops(edge_index, num_nodes):  # from pyg
    mask = edge_index[0] != edge_index[1]
    all_nodes = torch.arange(0, num_nodes, dtype=edge_index[0].dtype, device=edge_index[0].device)
    return torch.cat([edge_index[:, mask], torch.stack((all_nodes, all_nodes))], dim=1)

# 对称归一化邻接矩阵的函数
def sym_normalization(edge_index, num_nodes, faster_device='cuda:0'):
    if num_nodes>1000000:  # adjust with GPU
        faster_device = 'cpu'
    original_device = edge_index.device
    begin = datetime.datetime.now()
    # 添加自环边
    edge_index = add_self_loops(edge_index, num_nodes)
    # 将边索引转换为稀疏 COO 格式的张量
    A = torch.sparse_coo_tensor(edge_index, torch.ones(len(edge_index[0])), (num_nodes, num_nodes), device=faster_device).coalesce()
    # 计算度的倒数
    degree_vec = torch.sparse.sum(A, 0).pow(-0.5).to_dense()
    # 创建对角矩阵
    I_edge_index = torch.stack((torch.arange(num_nodes), torch.arange(num_nodes)))
    D_rsqrt = torch.sparse_coo_tensor(I_edge_index, degree_vec, (num_nodes, num_nodes), device=faster_device)
    # 对称归一化邻接矩阵
    DA = torch.sparse.mm(D_rsqrt, A)
    del A  # to save GPU mem  释放GPU内存
    DAD = torch.sparse.mm(DA, D_rsqrt)
    del DA
    end = datetime.datetime.now()
    print('sym norm done',  end - begin)
    return DAD.coalesce().to(original_device)



# 将稀疏张量按指定维度分割
def sparse_2d_split(st, split_size, split_dim=0):
    seps = list(range(0, st.size(split_dim), split_size)) + [st.size(split_dim)]
    parts = []
    split_idx = st.indices()[split_dim]
    other_idx = st.indices()[1 - split_dim]
    def make_2d_st(idx0, idx1, val, sz0, sz1):
        return  torch.sparse_coo_tensor(torch.stack([idx0, idx1]), val, (sz0, sz1)).coalesce()
    for lower, upper in zip(seps[:-1], seps[1:]):
        mask: torch.Tensor = (split_idx < upper) & (split_idx >= lower)
        if split_dim == 0:
            parts.append(make_2d_st(split_idx[mask]-lower, other_idx[mask], st.values()[mask], upper-lower, st.size(1)))
        else:
            parts.append(make_2d_st(other_idx[mask], split_idx[mask]-lower, st.values()[mask], st.size(0), upper-lower))
    return parts


def sparse_2d_split_csr(st, split_size, split_dim=0):
    seps = list(range(0, st.size(split_dim), split_size)) + [st.size(split_dim)]
    print(seps)
    print(st)
    print(st.col_indices().shape)
    print(st.shape)
    parts = []
    split_idx_row = st.crow_indices()
    print("st.crow_indices()[1]", int(st.crow_indices()[1]))
    split_idx_col = st.col_indices()
    def make_2d_st(idx0, idx1, val, sz0, sz1):
        print("idx0",idx0.shape,"idx1",idx1.shape,"val",val.shape,"sz0",sz0,"sz1",sz1)
        return  torch.sparse_csr_tensor(idx0, idx1, val, (sz0, sz1))
    for lower, upper in zip(seps[:-1], seps[1:]):
        print("upper",upper,"lower",lower)
        mask_row: torch.Tensor = (torch.arange(len(split_idx_row)) < upper + 1) & (torch.arange(len(split_idx_row)) >= lower)
        print("split_idx_row[mask_row].shape",split_idx_row[mask_row].shape)
        col_upper = int(split_idx_row[upper])
        col_lower = int(split_idx_row[lower])
        print("col_upper",col_upper,"col_lower",col_lower)
        mask_col: torch.Tensor = (torch.arange(len(split_idx_col)) < col_upper) & (torch.arange(len(split_idx_col)) >= col_lower)
        parts.append(make_2d_st(split_idx_row[mask_row]-col_lower, split_idx_col[mask_col], st.values()[mask_col], upper-lower, st.size(1)))
    return parts