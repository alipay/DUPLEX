import os
import sys
sys.path.append('../')
import dgl
import pandas as pd
import numpy as np
import torch
from duplex.mylogging import *
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import pdb
import networkx as nx
try:
    from stellargraph.data import EdgeSplitter
except:
    print('no stellargraph')
import re
from duplex.utils import undirected_label2directed_label
import random

import scipy.sparse as sp
def load_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle = True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        z = loader.get('labels')

        graph = {
            'A': A,
            'X': X,
            'z': z
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph

def generate_random_initialize(filename, num_nodes, dim):
    print('Genarating random initial...')
    embedding_matrix = torch.tensor(np.random.rand(num_nodes, dim).astype(np.float32))
    np.savetxt(filename, embedding_matrix, fmt='%f', delimiter=',')
    return embedding_matrix

def read_initialize(filename):
    try:
        embedding_matrix = torch.tensor(np.loadtxt(filename, dtype=float, delimiter=' ').astype(np.float32))
    except:
        embedding_matrix = torch.tensor(np.loadtxt(filename, dtype=float, delimiter=',').astype(np.float32))
    return embedding_matrix

# args, nodes, labels, '../data/citeseer/', seed, 'cpu'
def split_data(args, nodes, labels, save_path, seed):
    """
    Split data into training, testing, and validation sets and save them as text files.

    Args:
    - args (Namespace): Arguments containing necessary parameters.
    - nodes (torch.Tensor): Tensor containing node IDs.
    - labels (torch.Tensor): Tensor containing node labels.
    - save_path (str): Path to save the split data.
    - seed (int): Seed for random shuffling.

    Returns:
    - str: Confirmation message indicating the process is done.
    """
    
    assert len(nodes)==len(labels)
    num_nodes = len(nodes)
    test_r, val_r = int(num_nodes*(args.test_val_ratio[0])), int(num_nodes*args.test_val_ratio[1])
    rand_idx = list(range(num_nodes))
    random.shuffle(rand_idx)
    
    test_nodes = torch.zeros((test_r,2))
    test_nodes[:,0] = nodes[rand_idx[:test_r]]
    test_nodes[:,1] = labels[rand_idx[:test_r]]
    val_nodes = torch.zeros((val_r,2))
    val_nodes[:,0] = nodes[rand_idx[test_r:test_r+val_r]]
    val_nodes[:,1] = labels[rand_idx[test_r:test_r+val_r]]
    train_nodes = torch.zeros((num_nodes-test_r-val_r,2))
    train_nodes[:,0] = nodes[rand_idx[test_r+val_r:]]
    train_nodes[:,1] = labels[rand_idx[test_r+val_r:]]
    print("train %s test %s val %s"%(num_nodes-test_r-val_r, test_r, val_r))
    # whole_graph
    np.savetxt("%s/train_nodes_%s.txt"%(save_path, seed), train_nodes, delimiter=',',fmt='%i')
    np.savetxt("%s/test_nodes_%s.txt"%(save_path, seed), test_nodes, delimiter=',', fmt='%i')
    np.savetxt("%s/val_nodes_%s.txt"%(save_path, seed), val_nodes, delimiter=',', fmt='%i')
    return 'Done'


def unique(x, dim=None):
    """
    Returns the unique elements of x along with the indices of those unique elements.

    Parameters:
    - x: Input tensor.
    - dim: Dimension along which to compute uniqueness. If None, the uniqueness is computed over the entire tensor.

    Returns:
    - unique: Tensor containing the unique elements of x.
    - inverse: Indices of the unique elements in the original tensor x.

    Reference:
    - https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

def get_train_edges(graph_file, save_path):
    """
    Extract training edges from the graph file and save them as a text file.

    Args:
    - graph_file (str): Path to the graph file.
    - save_path (str): Path to save the extracted training edges.
    """
    
    graph_file = graph_file
    glist, _ = dgl.data.utils.load_graphs(graph_file)
    graph = glist[0]
    A = graph.adj()

    train_eid = torch.range(0,graph.num_edges()-1) # all edges on the graph
    bi_graph = dgl.add_edges(graph, graph.reverse().edges()[0], graph.reverse().edges()[1])
    neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(2)
    neg_edges = neg_sampler(bi_graph, train_eid) 
    
    train_src = graph.edges()[0]
    train_dst = graph.edges()[1]
    train_labels = torch.ones(train_dst.shape)
    train_edges = torch.hstack((train_src.reshape(-1,1),train_dst.reshape(-1,1),train_labels.reshape(-1,1))) 

    neg_train_edges = torch.hstack((neg_edges[0].reshape(-1,1),neg_edges[1].reshape(-1,1),torch.zeros(neg_edges[0].shape).reshape(-1,1)))
    di_train_edges = torch.vstack((train_edges, neg_train_edges))
    di_train_edges, di_labels_train = undirected_label2directed_label(A, di_train_edges, 4, 1) # same label with above

    np.savetxt(save_path+'/train_4.txt', di_train_edges, delimiter=',', fmt='%i')

def load_feature_data(args, nodes):
    file_name = args.initial_path+'features.txt'
    print('load features')
    embedding_matrix = read_initialize(file_name)
    return embedding_matrix

def load_graph_data_transductive(args):
    """
    Load and preprocess graph data for transductive learning.

    Args:
    - args (Namespace): Arguments containing necessary parameters.

    Returns:
    - train_dataloader (dgl.dataloading.DataLoader): DataLoader for training samples.
    - whole_graph (dgl.DGLGraph): The whole graph.
    - embedding_matrix (torch.Tensor): Embedding matrix for nodes.
    """
    
    graph_file = args.training_path+'whole.graph'
    glist, _ = dgl.data.utils.load_graphs(graph_file)
    whole_graph = glist[0]
    nodes = whole_graph.num_nodes()
    # ---------- graph ------
    whole_graph.edata['exist'] = torch.tensor([1.]*whole_graph.num_edges())
    whole_graph = dgl.add_edges(whole_graph, whole_graph.reverse().edges()[0], whole_graph.reverse().edges()[1]) 
    whole_graph.edata['exist'][whole_graph.edata['exist'] == 0] = -1.
    whole_graph.edata['am_exist']= torch.tensor([0.]*whole_graph.num_edges())
    whole_graph.edata['am_exist'][whole_graph.edata['exist'] != 0] = 1.

    # Remove duplicate edges
    w_edges = torch.cat((whole_graph.edges()[0].unsqueeze(1),whole_graph.edges()[1].unsqueeze(1)),dim=1)
    _, idx = unique(w_edges,dim=0)
    dup = (torch.arange(len(w_edges)).unsqueeze(1) != idx).all(1)
    whole_graph.edata['am_exist'][dup] = 0.

    # Initialize node features using the embedding matrix
    embedding_matrix = load_feature_data(args, nodes)
    whole_graph.ndata['h'] = torch.tensor(embedding_matrix)
    print('test if zero ',embedding_matrix.sum(dim=-1)[:5])
    logging.info('shape of graph embedding(%s,%s)'%(whole_graph.ndata['h'].shape))
    
    file_label = args.initial_path+'labels.txt'
    if os.path.exists(file_label):
        labels = np.loadtxt(file_label, dtype=int)
        logging.info('label num: %s'%(str(np.unique(labels, return_counts=True))))
    else:
        print('no label file')
    whole_graph.ndata['label'] = torch.tensor(labels)
    
    # Load and process training samples
    train_file = '/'.join((args.training_path, 'train_nodes_'+str(args.seed)+'.txt'))
    if os.path.exists(train_file):
        train_samples = torch.tensor(np.loadtxt(train_file, dtype=int, delimiter=','))
    else: 
        print('no train_file')
        
    # Balance training samples based on class distribution if specified    
    if args.balance>0:
        train_labels = train_samples[:,1]
        train_indices = get_train_val_test_split(train_labels, train_examples_per_class=args.balance)
        train_samples = train_samples[train_indices]

    print('train_samples:',str(train_samples[:,1].unique(return_counts=True)))
    train_nids = train_samples[:,0]

    # Create a MultiLayerFullNeighborSampler for DataLoader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    train_dataloader = dgl.dataloading.DataLoader(
        whole_graph, train_nids, sampler,  
        batch_size=args.bc_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=args.num_workers,
        device=args.device,
        use_uva = args.use_uva)

    return train_dataloader, whole_graph, embedding_matrix


def load_graph_data_inductive(args):
    """
    Load and preprocess graph data for inductive learning.

    Args:
    - args (Namespace): Arguments containing necessary parameters.

    Returns:
    - train_dataloader (dgl.dataloading.DataLoader): DataLoader for training samples.
    - whole_graph (dgl.DGLGraph): The whole graph.
    - embedding_matrix (torch.Tensor): Embedding matrix for nodes.
    - val_blocks (List[dgl.dataloading.Block]): List of blocks for validation.

    Note:
    This function performs various preprocessing steps such as creating train-validation splits, initializing node features, and setting up the DataLoader for training samples.
    """
    
    graph_file = args.training_path+'whole.graph'
    glist, _ = dgl.data.utils.load_graphs(graph_file)
    whole_graph = glist[0]
    nodes = whole_graph.num_nodes()
    # ------- train ------
    train_file = '/'.join((args.training_path, 'train_nodes_'+str(args.seed)+'.txt'))
    val_file = '/'.join((args.training_path, 'val_nodes_'+str(args.seed)+'.txt'))
    if os.path.exists(train_file):
        train_samples = torch.tensor(np.loadtxt(train_file, dtype=int, delimiter=','))
        val_samples = torch.tensor(np.loadtxt(val_file, dtype=int, delimiter=','))
    else: 
        print('no train_file')
    
    if args.balance>0:
        train_labels = train_samples[:,1]
        train_indices = get_train_val_test_split(train_labels, train_examples_per_class=args.balance)
        train_samples = train_samples[train_indices]

    print('train_samples:',str(train_samples[:,1].unique(return_counts=True)))
    train_nids = train_samples[:,0]
    train_val_nids = torch.cat((train_nids, val_samples[:,0]),dim=0)
    train_val_graph = dgl.node_subgraph(whole_graph, train_val_nids) # examples : 0, 5, 10, ...
    train_val_nodes = train_val_graph.ndata[dgl.NID] # 0,1,2,3,4,5,...
    # ---------- train graph ------
    train_val_graph.edata['exist'] = torch.tensor([1.]*train_val_graph.num_edges())
    train_val_graph = dgl.add_edges(train_val_graph, train_val_graph.reverse().edges()[0], train_val_graph.reverse().edges()[1]) 
    train_val_graph.edata['exist'][train_val_graph.edata['exist'] == 0] = -1.
    train_val_graph.edata['am_exist']= torch.tensor([0.]*train_val_graph.num_edges())
    train_val_graph.edata['am_exist'][train_val_graph.edata['exist'] != 0] = 1.

    # Initialize node features using the embedding matrix
    embedding_matrix = load_feature_data(args, nodes)
    print('test if zero ',embedding_matrix.sum(dim=-1)[:5], 'emb file shape: ', embedding_matrix.shape)
    train_val_graph.ndata['h'] = torch.tensor(embedding_matrix)[train_val_nodes]
    logging.info('shape of graph embedding(%s,%s)'%(train_val_graph.ndata['h'].shape))
    
    file_label = args.initial_path+'labels.txt'
    if os.path.exists(file_label):
        labels = np.loadtxt(file_label, dtype=int)
        logging.info('label num: %s'%(str(np.unique(labels, return_counts=True))))
    else:
        print('no label file')
    train_val_graph.ndata['label'] = torch.tensor(labels)[train_val_nodes]

    old2new = torch.zeros(max(train_val_nodes)+1,dtype=int) # node ids of train_val_graph
    old2new[train_val_nodes]=torch.range(0, train_val_nodes.shape[0]-1,dtype=int)
    train_nids = old2new[train_nids]

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    if args.use_model in ['DUPLEX_gat']:
        train_val_graph = dgl.add_self_loop(train_val_graph)
    train_dataloader = dgl.dataloading.DataLoader(
        train_val_graph, train_nids, sampler,  
        batch_size=args.bc_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=args.num_workers,
        device=args.device,
        use_uva = args.use_uva)
    
    val_nodes = val_samples[:,0]
    val_nodes = old2new[val_nodes]
    val_input_nodes, val_output_nodes, val_blocks = sampler.sample_blocks(train_val_graph.to(args.device), val_nodes.to(args.device))

    # Initialize node features and edge features for the whole graph
    whole_graph.edata['exist'] = torch.tensor([1.]*whole_graph.num_edges())
    whole_graph = dgl.add_edges(whole_graph, whole_graph.reverse().edges()[0], whole_graph.reverse().edges()[1]) 
    whole_graph.edata['exist'][whole_graph.edata['exist'] == 0] = -1.
    whole_graph.edata['am_exist']= torch.tensor([0.]*whole_graph.num_edges())
    whole_graph.edata['am_exist'][whole_graph.edata['exist'] != 0] = 1.
    whole_graph.ndata['h'] = torch.tensor(embedding_matrix)
    whole_graph.ndata['label'] = torch.tensor(labels)
    if args.use_model in ['DUPLEX_gat']:
        whole_graph = dgl.add_self_loop(whole_graph)

    return train_dataloader, whole_graph, embedding_matrix, val_blocks

def load_train_test_data(args, whole_graph):
    """
    Load and preprocess training and testing data.

    Args:
    - args (Namespace): Arguments containing necessary parameters.
    - whole_graph (dgl.DGLGraph): The whole graph.

    Returns:
    - train_samples (torch.Tensor): Tensor containing training samples.
    - val_samples (torch.Tensor): Tensor containing validation samples.
    - val_blocks (List[dgl.dataloading.Block]): List of blocks for validation.
    - test_samples (torch.Tensor): Tensor containing test samples.
    - test_blocks (List[dgl.dataloading.Block]): List of blocks for testing.
    """
    
    whole_graph = whole_graph.to(args.device)
    train_file = '/'.join((args.training_path, 'train_nodes_'+str(args.seed)+'.txt'))
    if os.path.exists(train_file):
        train_samples = torch.tensor(np.loadtxt(train_file, dtype=int, delimiter=',')).to(args.device)
    else: 
        print('no train_file')
    if args.balance>0:
        train_labels = train_samples[:,1]
        train_indices = get_train_val_test_split(train_labels, train_examples_per_class=args.balance)
        train_samples = train_samples[train_indices]

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    test_file = '/'.join((args.training_path, 'test_nodes_'+str(args.seed)+'.txt'))
    if os.path.exists(test_file):
        test_samples = torch.tensor(np.loadtxt(test_file, dtype=int, delimiter=',')).to(args.device)
    else: 
        print('no test_file')
    test_nodes = test_samples[:,0].unique()
    test_input_nodes, test_output_nodes, test_blocks = sampler.sample_blocks(whole_graph, test_nodes)

    val_file = '/'.join((args.training_path, 'val_nodes_'+str(args.seed)+'.txt'))
    if os.path.exists(val_file):
        val_samples = torch.tensor(np.loadtxt(val_file, dtype=int, delimiter=',')).to(args.device)
    else: 
        print('no val_file')
    
    val_nodes = val_samples[:,0].unique()
    val_input_nodes, val_output_nodes, val_blocks = sampler.sample_blocks(whole_graph, val_nodes)

    return train_samples, val_samples, val_blocks, test_samples, test_blocks

def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    '''
    Reference: https://github.com/matthew-hirn/magnet
    '''
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    print(num_samples,num_classes)
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples


    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    '''
    Reference: https://github.com/matthew-hirn/magnet
    '''
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size, val_size, test_size)

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    '''
    Reference: https://github.com/matthew-hirn/magnet
    '''
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])