# Copyright 2021, Zhao CHEN
# All rights reserved.
import coo_graph
import argparse


def main():
    cached = True
    r = coo_graph.COO_Graph_Full('reddit')
    # r = coo_graph.COO_Graph('cora')
    # r = coo_graph.COO_Graph('test', full_graph_cache_enabled=cached)
    # r = coo_graph.COO_Graph('flickr', full_graph_cache_enabled=cached)
    # r = coo_graph.COO_Graph('reddit', full_graph_cache_enabled=cached)
    # r = coo_graph.COO_Graph('ogbn-arxiv', full_graph_cache_enabled=cached)
    # r.partition(8)
    r.partition(2)
    return
    # for name in ['amazon-products', 'ogbn-products']:
    # for name in ['ogbn-arxiv', 'ogbn-products']:
    #     r = coo_graph.COO_Graph(name, full_graph_cache_enabled=cached)
    #     r.partition(4)
    #     r.partition(8)
    #     print(r)
    # return
    # for name in ['reddit', 'yelp', 'flickr', 'cora', 'ogbn-arxiv']:
    #     r = coo_graph.COO_Graph(name, full_graph_cache_enabled=cached)
    #     r.partition(8)
    #     r.partition(4)
    #     print(r)
    # return


if __name__ == '__main__':
    main()
