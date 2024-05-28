import dgl
import pandas as pd
import numpy as np
import os
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
from duplex.utils import undirected_label2directed_label

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
    embedding_matrix = np.random.normal(0,1,(num_nodes, dim*2)).astype(np.float32)
    np.savetxt(filename, embedding_matrix, fmt='%f', delimiter=',')
    return embedding_matrix

def read_initialize(filename):
    embedding_matrix = np.loadtxt(filename, dtype=float, delimiter=',').astype(np.float32)
    return embedding_matrix

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
    rand_idx = torch.randint(0, num_nodes,(num_nodes,))
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
    train_edges = torch.hstack((train_src.reshape(-1,1),train_dst.reshape(-1,1),train_labels.reshape(-1,1))) # non-existent

    neg_train_edges = torch.hstack((neg_edges[0].reshape(-1,1),neg_edges[1].reshape(-1,1),torch.zeros(neg_edges[0].shape).reshape(-1,1)))
    di_train_edges = torch.vstack((train_edges, neg_train_edges))
    di_train_edges, di_labels_train = undirected_label2directed_label(A, di_train_edges, 4, 1) # same label with above

    np.savetxt(save_path+'/train_4.txt', di_train_edges, delimiter=',', fmt='%i')

def load_graph_data(args):
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
    all_nodes = whole_graph.nodes()
    nodes = whole_graph.num_nodes()

    # Initialize node features using the embedding matrix
    file_name = args.initial_path+'initialize.txt'
    print('load random initialize')
    if os.path.exists(file_name):
        embedding_matrix = read_initialize(file_name)
        if embedding_matrix.shape!=(nodes,args.input_dim):
            embedding_matrix = generate_random_initialize(file_name, nodes, args.input_dim)
    else:
        if not os.path.exists(args.initial_path):
            os.makedirs(args.initial_path)
        embedding_matrix = generate_random_initialize(file_name, nodes, args.input_dim)
    
    whole_graph.ndata['h'] = torch.tensor(embedding_matrix)
    logging.info('shape of graph embedding(%s,%s)'%(whole_graph.ndata['h'].shape))

    # Load and process training samples
    training_file = args.training_path+'train_4.txt'
    train_edges = torch.tensor(np.loadtxt(training_file, dtype=int, delimiter=','))
    true_train_edges = train_edges[(train_edges[:,2]==1)|(train_edges[:,2]==2)]
    rev_train_edges = train_edges[(train_edges[:,2]==0)]
    none_edges_train = train_edges[(train_edges[:,2]==3)]

    pos_eid_train =  whole_graph.edge_ids(true_train_edges[:,0], true_train_edges[:,1])
    rev_eid_train = whole_graph.edge_ids(rev_train_edges[:,1], rev_train_edges[:,0])
    
    # Set edge labels (1,2) and existence indicators for training edges
    whole_graph.edata['label'] = torch.tensor([-1]*whole_graph.num_edges())
    whole_graph.edata['label'][pos_eid_train] = true_train_edges[:,2]
    whole_graph.edata['label'][rev_eid_train] = rev_train_edges[:,2]

    whole_graph.edata['exist'] = torch.tensor([1.]*whole_graph.num_edges())
    whole_graph = dgl.add_edges(whole_graph, whole_graph.reverse().edges()[0], whole_graph.reverse().edges()[1]) 
    whole_graph.edata['exist'][whole_graph.edata['exist'] == 0] = -1.

    # Add negative edges for training    
    whole_graph = dgl.add_edges(whole_graph, none_edges_train[:,0], none_edges_train[:,1])
    none_eid_train = whole_graph.edge_ids(none_edges_train[:,0], none_edges_train[:,1])
    whole_graph.edata['exist'][none_eid_train] = 0.
    whole_graph.edata['am_exist']= torch.tensor([0.]*whole_graph.num_edges())
    whole_graph.edata['am_exist'][whole_graph.edata['exist'] != 0] = 1.

    # Remove duplicate edges from the graph    
    w_edges = torch.cat((whole_graph.edges()[0].unsqueeze(1),whole_graph.edges()[1].unsqueeze(1)),dim=1)
    _, idx = unique(w_edges,dim=0)
    dup = (torch.arange(len(w_edges)).unsqueeze(1) != idx).all(1)
    whole_graph.edata['am_exist'][dup] = 0.

    # Create DataLoader for edge prediction training   
    whole_graph.edata['label'][none_eid_train] = none_edges_train[:,2]
    train_eid = torch.cat((pos_eid_train, rev_eid_train, none_eid_train))
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers))
    
    if args.use_model in ['DUPLEX_gat']:
        whole_graph = dgl.add_self_loop(whole_graph)
    dataloader = dgl.dataloading.DataLoader(
        whole_graph, train_eid, sampler,
        batch_size=args.bc_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=args.num_workers,
        device=args.device,
        use_uva = args.use_uva)

    sampler_nodes = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    all_dataloader = dgl.dataloading.DataLoader(
        whole_graph, all_nodes, sampler_nodes,
        batch_size=args.bc_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=args.num_workers,
        device=args.device,
        use_uva = args.use_uva)

    return dataloader, whole_graph, embedding_matrix, all_dataloader

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
    
    print('load data, seed ',args.seed)
    train_file = '/'.join((args.training_path, 'train_nodes_'+str(args.seed)+'.txt'))
    if os.path.exists(train_file):
        train_samples = torch.tensor(np.loadtxt(train_file, dtype=int, delimiter=','))
        print('train: ',str(train_samples[:,1].unique(return_counts=True)))

    else:
        print('no train_file')

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    test_file = '/'.join((args.training_path, 'test_nodes_'+str(args.seed)+'.txt'))
    if os.path.exists(test_file):
        test_samples = torch.tensor(np.loadtxt(test_file, dtype=int, delimiter=','))
        print('test: ',str(test_samples[:,1].unique(return_counts=True)))

    else:
        print('no test_file')
    test_nodes = test_samples[:,0].unique()
    test_input_nodes, test_output_nodes, test_blocks = sampler.sample_blocks(whole_graph, test_nodes)

    val_file = '/'.join((args.training_path, 'val_nodes_'+str(args.seed)+'.txt'))
    if os.path.exists(val_file):
        val_samples = torch.tensor(np.loadtxt(val_file, dtype=int, delimiter=','))
        print('val: ',str(val_samples[:,1].unique(return_counts=True)))

    else:
        print('no val_file')
    
    val_nodes = val_samples[:,0].unique()
    val_input_nodes, val_output_nodes, val_blocks = sampler.sample_blocks(whole_graph, val_nodes)

    return train_samples, val_samples, val_blocks, test_samples, test_blocks


