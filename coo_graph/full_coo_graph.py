import datetime
import os.path
import torch
import torch.sparse
from . import graph_utils
from . import datasets


class BasicGraph:
    def __init__(self, d, name, device):
        # 构造函数，初始化基本图属性
        self.name, self.device, self.attr_dict = name, device, d
        self.adj = d['adj']
        self.features = d['features']
        self.labels = d['labels'].to(device).to(torch.float if d['labels'].dim()==2 else torch.long)
        self.train_mask, self.val_mask, self.test_mask = (d[t].bool().to(device) for t in ("train_mask", 'val_mask', 'test_mask'))
        self.num_nodes, self.num_edges, self.num_classes = d["num_nodes"], d['num_edges'], d['num_classes']

    def __repr__(self):
        # 返回图的字符串表示
        masks = ','.join(str(torch.sum(mask).item()) for mask in [self.train_mask, self.val_mask, self.test_mask])
        return f'<COO Graph: {self.name}, |V|: {self.num_nodes}, |E|: {self.num_edges}, masks: {masks}>'


class GraphCache:
    # 图缓存类，提供图数据的保存和加载
    @staticmethod
    def full_graph_path(name, preprocess_for, root=datasets.data_root):
        return os.path.join(root, f'{name}_{preprocess_for}_full.coo_graph')
    @staticmethod
    def parted_graph_path(name, preprocess_for, rank, num_parts, root=datasets.data_root):
        dirpath = os.path.join(root, f'{name}_{preprocess_for}_{num_parts}_full')
        os.makedirs(dirpath, exist_ok=True)
        return os.path.join(dirpath, f'part_{rank}_of_{num_parts}.full_coo_graph')
    @staticmethod
    def save_dict(d, path):
        # 保存字典到文件
        if os.path.exists(path):
            print(f'warning: cache file {path} is overwritten.')
        d_to_save = {}
        for k, v in d.items():
            d_to_save[k] = v.clone() if type(v)==torch.Tensor else v
        torch.save(d_to_save, path)
    @staticmethod
    def load_dict(path):
        # 从文件加载字典
        d = torch.load(path)
        updated_d = {}
        for k, v in d.items():
            if type(v) == torch.Tensor and v.is_sparse:
                updated_d[k] = v.coalesce()
        d.update(updated_d)
        return d


class COO_Graph_Full(BasicGraph):
    def __init__(self, name, full_graph_cache_enabled=True, device='cpu', preprocess_for='GCN'):
        # 构造函数，初始化 COO 图
        self.preprocess_for = preprocess_for
        self.cache_path = GraphCache.full_graph_path(name, preprocess_for)
        if full_graph_cache_enabled and os.path.exists(self.cache_path):
            cached_attr_dict = GraphCache.load_dict(self.cache_path)
        else:
            src_data = datasets.load_dataset(name)
            cached_attr_dict = graph_utils.preprocess(name, src_data, preprocess_for)  # norm feat, remove edge_index, add adj
            GraphCache.save_dict(cached_attr_dict, self.cache_path)
        super().__init__(cached_attr_dict, name, device)

    def partition(self, num_parts, padding=True):
        # 对图进行分区
        begin = datetime.datetime.now()
        print(self.name, num_parts, 'partition begin', begin)
        attr_dict = self.attr_dict.copy()
        split_size = (self.num_nodes+num_parts-1)//num_parts
        pad_size = split_size*num_parts-self.num_nodes

        # 取消分区
        # adj_list = graph_utils.sparse_2d_split(self.adj, split_size)
        adj_list = self.adj
        features_list = list(torch.split(self.features, split_size))

        if padding and pad_size>0:
            # 如果启用填充，添加零填充以使每个部分大小相同
            padding_feat = torch.zeros((pad_size, self.features.size(1)), dtype=self.features.dtype, device=self.device)
            features_list[-1] = torch.cat((features_list[-1], padding_feat))

            padding_labels_size = torch.Size([pad_size])+self.labels.size()[1:]
            padding_labels = torch.zeros(padding_labels_size, dtype=self.labels.dtype, device=self.device)
            attr_dict['labels'] = torch.cat((self.labels, padding_labels))

            padding_mask = torch.zeros(pad_size, dtype=self.train_mask.dtype, device=self.device)
            for key in ['train_mask', 'val_mask', 'test_mask']:
                attr_dict[key] = torch.cat((attr_dict[key], padding_mask))

            adj_list = torch.sparse_coo_tensor(adj_list._indices(), adj_list._values(), (split_size*num_parts, split_size*num_parts))

        for i in range(num_parts):
            # 保存每个部分的图数据
            cache_path = GraphCache.parted_graph_path(self.name, self.preprocess_for, i, num_parts)
            attr_dict.update({'adj': adj_list, 'features': features_list[i]})
            GraphCache.save_dict(attr_dict, cache_path)
            Full_COO_Graph(self.name, i, num_parts, preprocess_for=self.preprocess_for)
        print(self.name, num_parts, 'partition done', datetime.datetime.now()-begin)


def coo_to_csr(coo, device, dtype):
    # 将 COO 格式的图转换为 CSR 格式
    print('coo', coo.size())
    csr = coo.to_sparse_csr()
    print('csr', csr.size())
    small_csr = torch.sparse_csr_tensor(csr.crow_indices().to(dtype=torch.int32), 
            csr.col_indices().to(dtype=torch.int32), csr.values().to(dtype=dtype), size=csr.size(), dtype=dtype, device=device)
    print('small csr', small_csr.size())
    return small_csr

class Full_COO_Graph(BasicGraph):
    def __init__(self, name, rank, num_parts, device='cpu', half_enabled=False, csr_enabled=False, preprocess_for='GCN'):
        # self.full_g = COO_Graph(name, preprocess_for, True, 'cpu')
        """
        初始化一个分区的 COO 图。

        Args:
            name (str): 图的名称。
            rank (int): 当前分区的等级或索引。
            num_parts (int): 分区的总数。
            device (str): 存储图数据的设备（默认为 'cpu'）。
            half_enabled (bool): 是否为特征启用半精度（float16）（默认为 False）。
            csr_enabled (bool): 是否使用 CSR 格式的邻接矩阵（默认为 False）。
            preprocess_for (str): 预处理的类型（默认为 'GCN'）。
        """
        # 继承 BasicGraph 类，初始化分区后的 COO 图
        self.rank, self.num_parts = rank, num_parts
        cache_path = GraphCache.parted_graph_path(name, preprocess_for, rank, num_parts)
        if not os.path.exists(cache_path):
            raise Exception('Not parted yet. Run COO_Graph.partition() first.', cache_path)
        # 加载缓存的属性字典
        cached_attr_dict = GraphCache.load_dict(cache_path)
        super().__init__(cached_attr_dict, name, device)

        # 本地图的属性
        self.local_num_nodes = self.adj.size(0)
        self.local_num_edges = self.adj.values().size(0)
        split_size = (self.local_num_nodes+num_parts-1)//num_parts
        self.local_labels = self.labels[split_size*rank:split_size*(rank+1)]
        self.local_train_mask = self.train_mask[split_size*rank:split_size*(rank+1)].bool()

        # adj and features are local already
        dtype=torch.float16 if half_enabled else torch.float
        self.features = self.features.to(device, dtype=dtype).contiguous()

        # 分割邻接矩阵

        # adj_parts = graph_utils.sparse_2d_split(self.adj, self.local_num_nodes, split_dim=1)
        adj_full = torch.sparse_coo_tensor(self.adj._indices(), self.adj._values(), (self.local_num_nodes, self.local_num_nodes))
        if csr_enabled:
            # 如果启用，将 COO 转换为 CSR 格式
            self.adj_full = coo_to_csr(adj_full, device, dtype)
        else:
            # 保持 COO 格式
            self.adj_full = adj_full.to(device=device, dtype=dtype) 

    def __repr__(self):
        # 返回图的字符串表示，包括分区信息
        local_g = f'<Local: {self.rank}, |V|: {self.local_num_nodes}, |E|: {self.local_num_edges}>'
        return super().__repr__() + local_g

