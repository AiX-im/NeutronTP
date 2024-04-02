import os
import torch
import torch.distributed as dist
import tempfile

from .timer import DistTimer
from .logger import DistLogger


class DistEnv:
    def __init__(self, rank, world_size, backend='nccl'):
        """
        分布式环境的初始化类

        Args:
            rank (int): 当前进程的等级或索引
            world_size (int): 分布式环境中的总进程数
            backend (str): 分布式通信后端，例如 'nccl'（默认为 'nccl'）。
        """
        assert(rank>=0)
        assert(world_size>0)
        self.rank, self.world_size = rank, world_size
        self.backend = backend
        self.init_device()
        self.init_dist_groups()
        self.logger = DistLogger(self)
        self.timer = DistTimer(self)
        self.store = dist.FileStore(os.path.join(tempfile.gettempdir(), 'torch-dist'), self.world_size)
        # for multi machine
        # self.store = dist.TCPStore("147.139.68.134", 29501, self.world_size , is_master=True)
        DistEnv.env = self  # no global...

    def __repr__(self):
        return '<DistEnv %d/%d %s>'%(self.rank, self.world_size, self.backend)

    def init_device(self):
        # 初始化设备，如果有多个 GPU，将设备设置为当前进程的 GPU 设备，否则为 CPU
        # if torch.cuda.device_count()>1:
        #     self.device = torch.device('cuda', self.rank)
        #     torch.cuda.set_device(self.device)
        # else:
        #     self.device = torch.device('cpu')
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda', 0)
        # torch.cuda.set_device(self.device)

    def all_reduce_sum(self, tensor):
        # 对所有进程中的张量进行求和操作，将结果存储在每个进程的张量中
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.world_group)

    def broadcast(self, tensor, src):
        # 从指定源进程广播张量到所有进程
        dist.broadcast(tensor, src=src, group=self.world_group)

    def all_gather_then_cat(self, src_t):
        # 在所有进程中收集张量，然后在维度0上进行拼接
        recv_list = [torch.zeros_like(src_t) for _ in range(self.world_size)]
        dist.all_gather(recv_list, src_t, group=self.world_group)
        return torch.cat(recv_list, dim=0)

    def barrier_all(self):
        # 对所有进程进行屏障同步，等待所有进程达到此点后再继续执行。
        dist.barrier(self.world_group)

    def init_dist_groups(self):
        # 初始化分布式通信组
        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size, init_method='env://')
        self.world_group = dist.new_group(list(range(self.world_size)))
        self.p2p_group_dict = {}
        for src in range(self.world_size):
            for dst in range(src+1, self.world_size):
                self.p2p_group_dict[(src, dst)] = dist.new_group([src, dst])
                self.p2p_group_dict[(dst, src)] = self.p2p_group_dict[(src, dst)]


class DistUtil:
    def __init__(self, env):
        self.env = env


if __name__ == '__main__':
    pass

