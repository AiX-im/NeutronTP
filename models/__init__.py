# Copyright 2021, Zhao CHEN
# All rights reserved.

from .gat import GAT
from .gcn import GCN
from .cached_gcn import CachedGCN
from .decouple_gcn import DecoupleGCN
from .tensplit_gcn import TensplitGCN
from .tensplit_gcn_large import TensplitGCNLARGE
from .tensplit_gcn_cpu import TensplitGCNCPU
from .tensplit_gat import TensplitGAT
from .tensplit_gcn_swap import TensplitGCNSWAP

def main():
    pass


if __name__ == '__main__':
    main()
