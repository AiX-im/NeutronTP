import os
import argparse
import torch

import dist_utils
import dist_train
import torch.distributed as dist


# 定义一个函数，用于包装分布式训练的设置和启动
def process_wrapper(rank, args, func):
    # for single machine
    # os.environ['MASTER_ADDR'] = '127.0.0.1'  #设置分布式训练的主节点，127.0.0.1默认是本地节点
    # os.environ['MASTER_PORT'] = '29500'    #端口号
    # os.environ['NCCL_SOCKET_IFNAME'] = 'lo' #NCCL网络接口名称，'lo'通常表示本地回环接口，GPU通信将通过本地主机进行
    # for multi machine
    os.environ['MASTER_ADDR'] = '202.199.6.34'  #设置分布式训练的主节点，127.0.0.1默认是本地节点
    os.environ['MASTER_PORT'] = '29500'    #端口号
    os.environ['NCCL_SOCKET_IFNAME'] = 'eno' #通信接口
    # os.environ['NCCL_SOCKET_IFNAME'] = 'lo' #NCCL网络接口名称，'lo'通常表示本地回环接口，GPU通信将通过本地主机进行
    # os.environ['NCCL_DEBUG']='INFO'
    # os.environ['NCCL_DEBUG_SUBSYS']='ALL'
    # os.environ['NCCL_P2P_DISABLE']='1'
    # os.environ['NCCL_ALGO'] = 'Ring'
    # os.environ['NCCL_MIN_NCHANNELS'] = '1'
    # os.environ['NCCL_MAX_NCHANNELS'] = '1'

    # 创建 DistEnv 对象，该对象封装了分布式训练的环境信息
    rank = 0 #多机下，要手动指定rank值
    env = dist_utils.DistEnv(rank, args.nnodes, args.backend)
    env.half_enabled = True
    env.csr_enabled = True

    # 调用传入的 func 函数，开始分布式训练
    func(env, args)


if __name__ == "__main__":
    num_GPUs = torch.cuda.device_count()
    parser = argparse.ArgumentParser() 
    # parser.add_argument("--nprocs", type=int, default=num_GPUs if num_GPUs>1 else 8)
    #single GPU
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=2)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--backend", type=str, default='nccl') #if num_GPUs>1 else 'gloo' 
    parser.add_argument("--dataset", type=str, default='reddit')
    parser.add_argument("--model", type=str, default='TensplitGCN')
    args = parser.parse_args()
    process_args = (args, dist_train.main)
    # 启动多个进程进行分布式训练
    torch.multiprocessing.spawn(process_wrapper, process_args, args.nprocs)
