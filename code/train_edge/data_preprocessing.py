import os
import sys
sys.path.append('../')
import dgl
import pandas as pd
import numpy as np
import torch
from duplex.utils import undirected_label2directed_label
from duplex.mylogging import *
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import pdb
import networkx as nx
try:
    from stellargraph.data import EdgeSplitter
except:
    print('no stellargraph')


def generate_random_initialize(filename, num_nodes, dim):
    embedding_matrix = np.random.normal(0,1,(num_nodes, dim*2)).astype(np.float32)
    np.savetxt(filename, embedding_matrix, fmt='%f', delimiter=',')
    return embedding_matrix

def read_initialize(filename):
    embedding_matrix = np.loadtxt(filename, dtype=float, delimiter=',').astype(np.float32)
    return embedding_matrix

def split_data(args, graph, save_path, seed, task):
    '''
    Split the input graph into training, testing, and validation edges with labels.

    Parameters:
    - args: Arguments from the command line.
    - graph: The input graph to be split.
    - save_path: The directory path where the split data will be saved.
    - seed: Random seed for reproducibility.
    - task: Integer indicating the type of task (1, 2, 3, or 4).

    Returns:
    None, saves the split data to files.

    Steps:
    1. Save the whole graph if not already saved.
    2. Convert the graph to a scipy sparse matrix for manipulation.
    3. Split test edges and labels, and save them.
    4. Split validation edges and labels, and save them.
    5. Generate training edges with labels and save them.

    Note:
    - If task is in [1, 2, 3, 4], additional processing is done to convert undirected labels to directed labels.
    - Negative edges are sampled for training based on the positive edges.
    '''
    
    pn_ratio = 1.0 # Set the positive-negative ratio for edge sampling
    
    # Create directories if they don't exist
    if not os.path.exists(save_path+str(seed)):
        os.makedirs(save_path+str(seed))
    if not os.path.exists(save_path+'whole.graph.txt'):
        dgl.data.utils.save_graphs(save_path+'whole.graph', graph)
        edges = torch.zeros(graph.num_edges(),2)
        edges[:,0] = graph.edges()[0]
        edges[:,1] = graph.edges()[1]
        np.savetxt(save_path+'whole.graph.txt',edges, fmt='%i')
    
    # Convert the graph to a scipy sparse matrix
    A = graph.adj()
    row=A._indices()[0]
    col=A._indices()[1]
    data=A._values()
    shape=A.size()
    A_sp=sp.csr_matrix((data, (row, col)), shape=shape)
    
    G = nx.from_scipy_sparse_array(A_sp) # create an undirected graph based on the adjacency

# ---- test -----
    edge_splitter_test = EdgeSplitter(G)
    G_test, test_edges, test_labels = edge_splitter_test.train_test_split(p=float(args.test_val_ratio[0]), method="global", keep_connected=True, seed = seed)
    if task in [1,2,3,4]:
        test_edges = np.hstack((test_edges,test_labels.reshape(-1,1)))
        di_test_edges, _ = undirected_label2directed_label(A, torch.tensor(test_edges), task, pn_ratio)
        print('sampled test edges',np.unique(di_test_edges[:,2],return_counts=True))
        np.savetxt(save_path+str(seed)+'/test_{}.txt'.format(task), di_test_edges, delimiter=',', fmt='%i')

# --- val ----
    edge_splitter_val = EdgeSplitter(G_test)
    G_val, val_edges, val_labels = edge_splitter_val.train_test_split(p=float(args.test_val_ratio[1]), method="global", keep_connected=True, seed = seed)
    
    if task in [1,2,3,4]:
        val_edges = np.hstack((val_edges, val_labels.reshape(-1,1)))
        di_val_edges, _ = undirected_label2directed_label(A, torch.tensor(val_edges), task, pn_ratio)
        print('sampled validation edges',np.unique(di_val_edges[:,2],return_counts=True))
        np.savetxt(save_path+str(seed)+'/val_{}.txt'.format(task), di_val_edges, delimiter=',', fmt='%i')

# ---- train ----
    train_graph = dgl.from_networkx(G_val)
    train_src = train_graph.edges()[0]
    train_dst = train_graph.edges()[1]
    train_labels = torch.ones(train_dst.shape)
    train_edges = torch.hstack((train_src.reshape(-1,1),train_dst.reshape(-1,1),train_labels.reshape(-1,1))) # undirected edges

# undirected edges to directed edges
    if task in [1,2,3,4]:
        pos_train_edges, _ = undirected_label2directed_label(A, train_edges, 'train_di', pn_ratio)
        print('sampled train edges',np.unique(pos_train_edges[:,2],return_counts=True))
        np.savetxt(save_path+str(seed)+'/train_di.txt', pos_train_edges, delimiter=',', fmt='%i')
        
        train_eid = graph.edge_ids(pos_train_edges[:,0], pos_train_edges[:,1])
        bi_graph = dgl.add_edges(graph, graph.reverse().edges()[0], graph.reverse().edges()[1])
        neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(2)
        neg_edges = neg_sampler(bi_graph, train_eid) 
        neg_train_edges = torch.hstack((neg_edges[0].reshape(-1,1),neg_edges[1].reshape(-1,1),torch.zeros(neg_edges[0].shape).reshape(-1,1)))
        all_train_edges = torch.vstack((pos_train_edges, neg_train_edges))
        di_train_edges, di_labels_train = undirected_label2directed_label(A, all_train_edges, task, pn_ratio)
        print('sampled train edges',np.unique(di_train_edges[:,2],return_counts=True))
        np.savetxt(save_path+str(seed)+'/train_{}.txt'.format(task), di_train_edges, delimiter=',', fmt='%i')

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

def load_training_data(args):
    """
    Load and preprocess training data for graph embedding training.

    Parameters:
    - args: Command-line arguments and settings.

    Returns:
    - dataloader: DataLoader for training.
    - whole_graph: DGL graph object containing the entire graph data.
    - embedding_matrix: Initial embedding matrix for nodes.

    Steps:
    1. Load the entire graph and initialize embedding.
    2. Load training edges and preprocess them for training.
    3. Set edge labels and existence indicators for training edges.
    4. Create a DataLoader for edge prediction training.
    """

    # Load entire graph and initialize embedding
    graph_file = args.training_path+'whole.graph'
    glist, _ = dgl.data.utils.load_graphs(graph_file)
    whole_graph = glist[0]
    nodes = whole_graph.num_nodes()
    
    # Initialize embedding matrix
    file_name = args.initial_path+'initialize.txt'
    embedding_matrix = generate_random_initialize(file_name, nodes, args.input_dim)
    whole_graph.ndata['h'] = torch.tensor(embedding_matrix)
    logging.info('shape of graph embedding(%s,%s)'%(whole_graph.ndata['h'].shape))

    # Load and preprocess training edges-
    training_file = '/'.join((args.training_path+str(args.seed),'train_4.txt'))
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

    return dataloader, whole_graph, torch.tensor(embedding_matrix)

def load_testing_data(args, task, whole_graph):
    """
    Load and preprocess testing and validation data for graph embedding testing.

    Parameters:
    - args: Command-line arguments and settings.
    - task: Integer indicating the type of task.
    - whole_graph: DGL graph object containing the entire graph data.

    Returns:
    - test_blocks: Blocks for testing data.
    - new_test_edges: Testing edges for the specified task.
    - val_blocks: Blocks for validation data.
    - new_val_edges: Validation edges for the specified task.

    Steps:
    1. Load testing edges if the testing file exists.
    2. Sample blocks for testing nodes in the whole graph.
    3. Load validation edges if the validation file exists.
    4. Sample blocks for validation nodes in the whole graph.
    """

    # Load testing edges if the file exists
    testing_file = '/'.join((args.training_path+str(args.seed),'test_'+str(task)+'.txt'))
    if os.path.exists(testing_file):
        new_test_edges = torch.tensor(np.loadtxt(testing_file, dtype=int, delimiter=',')).to(args.device)
    else: 
        print('no testing file')

    # Sample blocks for testing nodes
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    test_nodes = new_test_edges[:,:2].unique()
    test_input_nodes, test_output_nodes, test_blocks = sampler.sample_blocks(whole_graph, test_nodes)

    # Load validation edges if the file exists
    val_file = '/'.join((args.training_path+str(args.seed),'val_'+str(task)+'.txt'))
    if os.path.exists(val_file):
        new_val_edges = torch.tensor(np.loadtxt(val_file, dtype=int, delimiter=',')).to(args.device)
    else: 
        print('no validation file')
        
    # Sample blocks for validation nodes
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    val_nodes = new_val_edges[:,:2].unique()
    val_input_nodes, val_output_nodes, val_blocks = sampler.sample_blocks(whole_graph, val_nodes)

    return test_blocks,new_test_edges, val_blocks,new_val_edges
