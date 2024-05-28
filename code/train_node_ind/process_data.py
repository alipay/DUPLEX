import os
import sys
sys.path.append('../')
import numpy as np
import dgl
import os
import torch
import data_preprocessing as dp
import argparse
from config import const_args
import pandas as pd
args = argparse.Namespace(**const_args)
import pdb

def ps_data(dataset):
    """
    Preprocess and save data for node classification.

    Args:
    - dataset (str): Name of the dataset.
    """
    
    print(os.listdir('../'))
    save_path = '../node_data/%s/'%(dataset)
    g = dp.load_dataset('../data/%s.npz'%(dataset))
    A, X, z = g['A'], g['X'], g['z']

    graph = dgl.from_scipy(A)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path+'whole.graph.txt'):
        dgl.data.utils.save_graphs(save_path+'whole.graph', graph)
        edges = torch.zeros(graph.num_edges(),3)
        edges[:,0] = graph.edges()[0]
        edges[:,1] = graph.edges()[1]
        edges[:,2] = torch.ones((graph.num_edges(),))
        np.savetxt(save_path+'whole.graph.txt',edges, fmt='%i')

    nodes = graph.nodes()
    labels = torch.tensor(g['z'])
    attris = torch.tensor(g['X'].todense())
    print(attris.shape)
    np.savetxt(save_path+'features.txt',attris, delimiter=',')
    np.savetxt(save_path+'labels.txt', labels,fmt='%i')

    for seed in range(10):
        dp.split_data(args, nodes, labels, save_path, seed)
    dp.get_train_edges(save_path+'whole.graph',save_path)
    
for dataset in ['citeseer','cora_ml']:
    print(dataset)
    ps_data(dataset)
