import data_preprocessing as dp
from config import const_args as args
import dgl
import argparse

args = argparse.Namespace(**args)

for dataset in ['citeseer','cora','epinions','twitter']:
    print(dataset)
    for task in [1,2,3,4]:
        for seed in range(10):
            graph = dgl.load_graphs('../edge_data/%s/whole.graph'%(dataset))[0][0]
            save_path = '../edge_data/%s/'%(dataset)
            dp.split_data(args, graph, save_path, seed, task)