# Copyright 2021, Zhao CHEN
# All rights reserved.

import os
import torch
# from .friendster import FriendSterDataset

# 设置数据的根目录、DGL 数据集目录和 PyTorch Geometric 数据集目录
data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
dgl_root = os.path.join(data_root, 'dgl_datasets')
pyg_root = os.path.join(data_root, 'pyg_datasets')
# 确保目录存在，如果不存在则创建
for path in [data_root, dgl_root, pyg_root]:
    os.makedirs(path, exist_ok=True)

# 保存图数据集的函数
def save_dataset(edge_index, features, labels, train_mask, val_mask, test_mask, num_nodes, num_edges, num_classes, name):
    if name.startswith('a_quarter'):
        # 如果数据集名字以 'a_quarter' 开头，保留图的四分之一节点数的数据
        max_node = num_nodes//4
        smaller_mask = (edge_index[0]<max_node) & (edge_index[1]<max_node)

        edge_index = edge_index[:, smaller_mask].clone()
        features = features[:max_node].clone()
        labels = labels[:max_node].clone()
        train_mask = train_mask[:max_node].clone()
        val_mask = val_mask[:max_node].clone()
        test_mask = test_mask[:max_node].clone()
        num_nodes = max_node
        num_edges = edge_index.size(1)
    # 构建保存路径
    path = os.path.join(data_root, name+'.torch')
    # 保存数据集至指定路径
    torch.save({"edge_index": edge_index.int(), "features": features, "labels": labels.char(),
                "train_mask": train_mask.bool(), 'val_mask': val_mask.bool(), 'test_mask': test_mask.bool(),
                "num_nodes": num_nodes, 'num_edges': num_edges, 'num_classes': num_classes}, path)

# 加载图数据集的函数
def load_dataset(name):
    # 构建加载路径
    path = os.path.join(data_root, name+'.torch')
    if not os.path.exists(path):
        # 如果文件不存在，调用 prepare_dataset(name) 准备数据集
        prepare_dataset(name)
    # 加载数据集
    return torch.load(path)

# DGL 数据集准备函数
def prepare_dgl_dataset(dgl_name, tag):
    import dgl
    # 定义数据集源
    dataset_sources = {'cora': dgl.data.CoraGraphDataset, 'reddit': dgl.data.RedditDataset}
    # 加载DGL数据集
    dgl_dataset: dgl.data.DGLDataset = dataset_sources[dgl_name](raw_dir=dgl_root)
    # 获取图对象
    g = dgl_dataset[0]
    # 将图的邻接矩阵转换为PyTorch张量
    edge_index = torch.stack(g.adj_sparse('coo'))
    print('dgl dataset', dgl_name, 'loaded')
    # 保存数据集至指定路径
    save_dataset(edge_index, g.ndata['feat'], g.ndata['label'],
                 g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'],
                 g.num_nodes(), g.num_edges(), dgl_dataset.num_classes, tag)


# DGL 数据集准备函数
def prepare_tsp_dataset(dgl_name, tag):
    # 定义数据集源
    dataset = FriendSterDataset()
    g = dataset[0]
    # 将图的邻接矩阵转换为PyTorch张量
    edge_index = torch.stack(g.adj_sparse('coo'))
    print('dgl dataset', dgl_name, 'loaded')
    # 保存数据集至指定路径
    save_dataset(edge_index, g.ndata['feat'], g.ndata['label'],
                 g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'],
                 g.num_nodes(), g.num_edges(), dataset.num_classes, tag)
    

# PyTorch Geometric 数据集准备函数
def prepare_pyg_dataset(pyg_name, tag):
    import torch_geometric
    import ogb.nodeproppred
    # 定义数据集源
    dataset_sources = {'reddit': torch_geometric.datasets.Reddit,
                       'flickr': torch_geometric.datasets.Flickr,
                       'yelp': torch_geometric.datasets.Yelp,
                        'amazon-products': torch_geometric.datasets.AmazonProducts,
                        }
    # 加载 PyTorch Geometric 数据集
    pyg_dataset: torch_geometric.data.Dataset = dataset_sources[pyg_name](root=os.path.join(pyg_root, pyg_name))
    print('pyg dataset', pyg_name, 'loaded')
    # 获取数据集中的第一个数据对象
    data: torch_geometric.data.Data = pyg_dataset[0]
    # 保存数据集至指定路径
    save_dataset(data.edge_index, data.x, data.y,
                 data.train_mask, data.val_mask, data.test_mask,
                 data.num_nodes, data.num_edges, pyg_dataset.num_classes, tag)


def prepare_ogb_dataset(pyg_name, tag):
    import torch_geometric
    import ogb.nodeproppred
    # 定义数据集源
    dataset_source =  ogb.nodeproppred.PygNodePropPredDataset
    #  dataset_sources = { 'ogbn-products', 'ogbn-arxiv', 'ogbn-papers100M', }
    # 加载 OGB 数据集
    dataset = dataset_source(root=os.path.join(pyg_root, pyg_name), name=pyg_name)
    print('ogb dataset', pyg_name, 'loaded')
    # 获取数据集中的第一个数据对象
    data: torch_geometric.data.Data = dataset[0]
    # 获取数据集的拆分索引
    split_idx = dataset.get_idx_split()
    # 创建bool掩码
    bool_mask = torch.zeros(data.num_nodes).bool()
    # 创建填充掩码的函数
    make_mask = lambda name: bool_mask.index_fill(0, split_idx[name], True)
    # 如果标签是 2D 的，将其转换为 1D
    if data.y.dim()==2 and data.y.size(1)==1:
        label_1d = torch.reshape(data.y, [-1])
    # 保存数据集至指定路径
    save_dataset(data.edge_index, data.x, label_1d,
                 make_mask('train'), make_mask('valid'), make_mask('test'),
                 data.num_nodes, data.num_edges, dataset.num_classes, tag)


# 选择并准备指定数据集的函数
def prepare_dataset(tag):
    if tag=='reddit':
        return prepare_pyg_dataset('reddit', tag)
    elif tag=='flickr':
        return prepare_pyg_dataset('flickr', tag)
    elif tag == 'yelp':  # graphsaints
        return prepare_pyg_dataset('yelp', tag)
    elif tag=='amazon-products':  # graphsaints
        return prepare_pyg_dataset('amazon-products', tag)
    elif tag=='cora':
        return prepare_dgl_dataset('cora', tag)
    elif tag=='reddit_reorder':
        return prepare_dgl_dataset('reddit', tag)
    elif tag=='a_quarter_reddit':
        return prepare_pyg_dataset('reddit', tag)
    elif tag=='ogbn-products':
        return prepare_ogb_dataset('ogbn-products', tag)
    elif tag == 'ogbn-arxiv':
        return prepare_ogb_dataset('ogbn-arxiv', tag)
    elif tag == 'ogbn-100m':
        return prepare_ogb_dataset('ogbn-papers100M', tag)
    elif tag == 'friendster':
        return prepare_tsp_dataset('friendster', tag)
    else:
        print('no such dataset', tag)


def check_edges(edge_index, num_nodes):
    print(f'edges {edge_index[0].size(0)} nodes:{num_nodes}')
    num_parts = 4
    split_size = num_nodes//num_parts
    first_limit = split_size
    last_limit = num_nodes - split_size

    fist_size = (edge_index[0] < first_limit).sum()
    last_size = (edge_index[0] > last_limit).sum()
    print(f'first block {fist_size} last block {last_size}')

    mask_first = (edge_index[0] < first_limit) & (edge_index[1] < first_limit)
    mask_last = (edge_index[0] > last_limit) & (edge_index[1] > last_limit)

    fist_size = edge_index[0][mask_first].size(0)
    last_size = edge_index[0][mask_last].size(0)
    print(f'first p {fist_size} last p {last_size}')


def main():
    prepare_dataset('ogbn-100m')
    return
    prepare_dataset('ogbn-arxiv')
    prepare_dataset('ogbn-products')
    return
    for dataset_name in ['cora', 'reddit', 'flickr', 'yelp', 'flickr', 'a_quarter_reddit','amazon-products']:
        prepare_dataset(dataset_name)
    return


if __name__ == '__main__':
    main()
