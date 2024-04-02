from coo_graph import Parted_COO_Graph
from coo_graph import Full_COO_Graph
from coo_graph import Full_COO_Graph_Large
from models import GCN, GAT, CachedGCN, DecoupleGCN, TensplitGCN, TensplitGCNLARGE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from sklearn.metrics import f1_score
from dist_utils import DistEnv


def f1(y_true, y_pred, multilabel=True):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    if multilabel:
        # 对预测值进行二值化
        y_pred[y_pred > 0.5] = 1.0
        y_pred[y_pred <= 0.5] = 0.0
        # 在一些节点（10、100、1000）上打印预测值和真实值
        for node in [10,100,1000]:
            DistEnv.env.logger.log('pred', y_pred[node] , rank=0)
            DistEnv.env.logger.log('true', y_true[node] , rank=0)
    else:
        # 如果是单标签分类，将预测值转为类别索引
        y_pred = np.argmax(y_pred, axis=1)
    # 返回 micro 和 macro 两种平均方式的 F1 分数
    return f1_score(y_true, y_pred, average="micro"), \
           f1_score(y_true, y_pred, average="macro")

def train(g, env, args):
    if args.model == 'GCN':
        model = GCN(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'CachedGCN':
        model = CachedGCN(g, env, hidden_dim=args.hidden)
    elif args.model == 'GAT':
        model = GAT(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'DecoupleGCN':
        model = DecoupleGCN(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'TensplitGCN':
        model = TensplitGCN(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'TensplitGCNLARGE':
        model = TensplitGCN(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)

    # 创建优化器（Adam）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if g.labels.dim()==1:
        # 对于单标签分类，使用交叉熵损失函数
        loss_func = nn.CrossEntropyLoss()
    elif g.labels.dim()==2:
        # 对于多标签分类，使用 BCEWithLogitsLoss 损失函数
        loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    for epoch in range(args.epoch):
        with env.timer.timing('epoch'):
            with autocast(env.half_enabled):
                # 前向传播，计算输出
                outputs = model(g.features)
                # 梯度清零
                optimizer.zero_grad()
                if g.local_labels[g.local_train_mask].size(0) > 0:
                    # 计算损失（仅在包含训练节点的分区上计算）
                    loss = loss_func(outputs[g.local_train_mask], g.local_labels[g.local_train_mask])
                else:
                    # 如果没有训练节点，输出警告并使用虚拟损失
                    env.logger.log('Warning: no training nodes in this partition! Backward fake loss.')
                    loss = (outputs * 0).sum()
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
            # 输出当前的损失信息
            env.logger.log("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()), rank=0)

        if epoch%10==0 or epoch==args.epoch-1:
            # 收集所有节点的输出，并拼接在一起
            all_outputs = env.all_gather_then_cat(outputs)
            if g.labels.dim()>1:
                # 如果是多标签分类，计算 F1 分数并打印
                mask = g.train_mask
                env.logger.log(f'Epoch: {epoch:03d}', f1(g.labels[mask], torch.sigmoid(all_outputs[mask])), rank=0)
            else:
                # 如果是单标签分类，计算并打印训练/验证/测试的准确率
                acc = lambda mask: all_outputs[mask].max(1)[1].eq(g.labels[mask]).sum().item()/mask.sum().item()
                env.logger.log(f'Epoch: {epoch:03d}, Train: {acc(g.train_mask):.4f}, Val: {acc(g.val_mask):.4f}, Test: {acc(g.test_mask):.4f}', rank=0)


def main(env, args):
    env.csr_enabled = False
    env.csr_enabled = True

    env.half_enabled = True
    env.half_enabled = False
    # 打印进程开始信息
    env.logger.log('proc begin:', env)
    with env.timer.timing('total'):
        # 使用 Parted_COO_Graph 加载分布式环境下的图数据
        if args.model == 'TensplitGCN':
            print(f"Rank: {env.rank}, world_size: {env.world_size}")
            g = Full_COO_Graph(args.dataset, env.rank, env.world_size, env.device, env.half_enabled, env.csr_enabled) #不再切分图邻接矩阵, 但feature按worker数均分
        elif args.model == 'TensplitGCNLARGE':
            g = Full_COO_Graph_Large(args.dataset, env.rank, env.world_size, env.device, env.half_enabled, env.csr_enabled) #保存feature与graph在CPU内存中
        else:
            g = Parted_COO_Graph(args.dataset, env.rank, env.world_size, env.device, env.half_enabled, env.csr_enabled)
        env.logger.log('graph loaded', g)
        env.logger.log('graph loaded\n', torch.cuda.memory_summary())
        # 调用 train 函数进行图神经网络训练
        train(g, env, args)
    # 打印model信息
    if env.rank == 0:    
        print(f"Model: {args.model} layers: {args.nlayers} dataset: {args.dataset} nprocs {args.nprocs}")
    # 打印计时器的总结信息
    env.logger.log(env.timer.summary_all(), rank=0)

