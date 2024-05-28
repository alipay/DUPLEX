import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from dgl.dataloading import Sampler
import dgl
from dgl.base import NID, EID, dgl_warning
from dgl.transforms import to_block
from dgl.dataloading.base import *
import pdb
from duplex.mylogging import *
from dgl.dataloading.dataloader import *
from dgl.dataloading.dataloader import _PrefetchingIter, _get_device
import pandas
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

def undirected_label2directed_label(adj, edge_pairs, task, ratio):
    """
    Convert undirected edge labels to directed edge labels based on the specified task and ratio.

    Parameters:
    - adj (torch.Tensor): Adjacency matrix representing the graph.
    - edge_pairs (torch.Tensor): Tensor containing edge pairs in the format (source, destination, label).
    - task (str or int): Specifies the task to perform. 'train_di' for training directed graphs, 0 for undirected existence,
      1 for existence prediction, 2 for direction prediction, 3 for three-type classification and 4 for four-type classification.
    - ratio (float): Ratio used for adjusting the balance between positive and negative edges.

    Returns:
    - new_edge_pairs (torch.Tensor): Filtered tensor containing directed edge pairs after label conversion.
    - new_labels (torch.Tensor): Corresponding labels for the filtered directed edge pairs.
    """
    
    new_edge_pairs = edge_pairs.clone()
    new_edge_pairs = new_edge_pairs.unique(dim=0)# no duplicates
    
    # Create a sparse tensor to mark the presence of edges
    values = torch.ones(len(new_edge_pairs))# select
    h = torch.sparse_coo_tensor(new_edge_pairs[:,:2].T, values, adj.shape)
    m1 = adj.mul(h)+adj.T.mul(h)
    m2 = adj.mul(h)-adj.T.mul(h)
    
    # here the order of edges no more exist
    type_1 = m1.coalesce().indices().T[m1.coalesce().values()==2.] # bidirected edge pairs(N,2)
    type_2 = m2.coalesce().indices().T[m2.coalesce().values()==1.] # positive edges
    type_3 = m2.coalesce().indices().T[m2.coalesce().values()==-1.] # reverse edges
    type_4 = new_edge_pairs[new_edge_pairs[:,2]==0][:,:2] # non-existent
    df = pandas.DataFrame({
        'src':type_1.min(dim=1).values.tolist()+type_2[:,0].tolist()+type_3[:,1].tolist()+type_4.min(dim=1).values.tolist(),
        'dst':type_1.max(dim=1).values.tolist()+type_2[:,1].tolist()+type_3[:,0].tolist()+type_4.max(dim=1).values.tolist(),
        'type':[2]*len(type_1)+[1]*len(type_2)+[1]*len(type_3)+[3]*len(type_4) # (min, max) and positive only
    })
    df2 = df.drop_duplicates(subset=['src','dst'])
    new_edge_pairs = torch.zeros((len(df2),3),dtype=int)
    new_edge_pairs[:,0] = torch.tensor(list(df2["src"]))
    new_edge_pairs[:,1] = torch.tensor(list(df2["dst"]))
    new_edge_pairs[:,2] = torch.tensor(list(df2["type"]))
    new_edge_pairs = new_edge_pairs.unique(dim=0)
    labels = new_edge_pairs[:,2]
    new_labels = labels.clone()
    
    # Adjust samples based on the task and ratio
    if task == 'train_di':
        # For training directed graphs, convert bidirectional edges to positive edges
        bi_edges = new_edge_pairs[new_edge_pairs[:,2] == 2].clone()
        bi_edges[:,2] = 1
        new_edge_pairs[:,2][new_edge_pairs[:,2] == 2] = 1
        bi_edges = torch.index_select(bi_edges, 1, torch.tensor([1,0,2]))
        new_edge_pairs = torch.cat((new_edge_pairs, bi_edges), dim=0)
        new_labels = new_edge_pairs[:,2].clone()
        new_labels[new_labels==0] = -1 
        new_labels[new_labels==3] = -1 
        
    elif task == 0: # undirected existence
        new_labels[labels == 3] = 0 # non-existent
        new_labels[labels == 2] = 1 # bi-directional
        pos_num = (new_labels == 1).sum()
        neg_num = (new_labels == 0).sum()

        if  ratio*pos_num > neg_num: # pos>neg
            pos = np.where(new_labels == 1)[0] # pos
            rng = np.random.default_rng(1)
            pos_half = rng.choice(pos, size= (pos_num-int(neg_num/ratio)).item(), replace=False) # balance
            new_labels[pos_half] = -1
        elif ratio*pos_num < neg_num: # neg>pos
            neg = np.where(new_labels == 0)[0] # neg
            rng = np.random.default_rng(1)
            neg_half = rng.choice(neg, size= (neg_num-int(ratio*pos_num)).item(), replace=False) # balance
            new_labels[neg_half] = -1

    elif task==1: # Existence Prediction
        # Randomly turn 1/3 (positive + bidirectional) of the positive edges into reverse edges, and randomly select 1/3 (positive + bidirectional) of non-existent edges.
        pos_num = (labels == 1).sum()
        neg_num = (labels == 3).sum()
        bi_num = (labels == 2).sum()
        new_labels[labels == 3] = -1
        pos = np.where(labels == 1)[0] # pos 
        rng = np.random.default_rng(1)
        pos_half = rng.choice(pos, size= int(1/3*(pos_num+bi_num)), replace=False)
        new_labels[pos_half] = 0
        src = new_edge_pairs[pos_half,0]
        dst = new_edge_pairs[pos_half,1]
        new_edge_pairs[pos_half,0] = dst
        new_edge_pairs[pos_half,1] = src

        neg = np.where(new_labels == -1)[0] # neg 
        rng = np.random.default_rng(1)
        neg_half = rng.choice(neg, size= int(1/3*(pos_num+bi_num)), replace=False) 
        new_labels[neg_half] = 0
        
        new_labels[labels == 2] = 1
    else:
        # step1: ensure the number of pos and neg edges
        pos_num = (labels == 1).sum()
        neg_num = (labels == 3).sum()
        bi_num = (labels == 2).sum()
        if  0.5*ratio*pos_num > neg_num: 
            pos = np.where(labels == 1)[0] 
            rng = np.random.default_rng(1)
            pos_half = rng.choice(pos, size= int(pos_num-int(neg_num/(0.5*ratio))), replace=False) 
            new_labels[pos_half] = -1
        elif 0.5*ratio*pos_num < neg_num: 
            neg = np.where(labels == 3)[0]
            rng = np.random.default_rng(1)
            neg_half = rng.choice(neg, size= int(neg_num-0.5*ratio*pos_num), replace=False)
            new_labels[neg_half] = -1
        if bi_num >= pos_num:
            bi = np.where(labels == 2)[0] # bi
            rng = np.random.default_rng(1)
            bi_half = rng.choice(bi, size= (bi_num-pos_num).item(), replace=False)
            new_labels[bi_half] = -1

        # step2: get reverse
        pos = np.where(labels == 1)[0]
        new_labels[pos[int(len(pos)/2):]] = 0
        s = new_edge_pairs[pos[int(len(pos)/2):],0]
        t = new_edge_pairs[pos[int(len(pos)/2):],1]
        new_edge_pairs[pos[int(len(pos)/2):],0] = t
        new_edge_pairs[pos[int(len(pos)/2):],1] = s
        
        if task == 3: # if three-type classification, delete 2
            new_labels[new_labels==2] = -1
        elif task == 2: # if direction prediction
            new_labels[(new_labels==2)|(new_labels==3)] = -1
    
    new_edge_pairs[:,2] = new_labels
    return new_edge_pairs[new_labels >= 0], new_labels[new_labels >= 0]

def edge_sampler(args, adj, pos_graph, neg_graph, task):
    '''
    Sample edges from positive and negative graphs of one batch using DGL.
    
    Args:
    - args (Namespace): Arguments containing the negative ratio.
    - adj (torch.Tensor): Adjacency matrix of the graph.
    - pos_graph (DGLGraph): Positive graph sampled by DGL.
    - neg_graph (DGLGraph): Negative graph sampled by DGL.
    - task (str or int): Specifies the task to perform.
    
    Returns:
    - new_edge_pairs (torch.Tensor): Filtered tensor containing directed edge pairs after label conversion.
    - labels (torch.Tensor): Corresponding labels for the filtered directed edge pairs.

    Note:
    The edges returned are all new edge IDs and need to be transformed. 
    Positive labels are denoted by 1, and negative labels are denoted by 0.
    '''
    
    psrc, pdst = pos_graph.edges() # return [x,...],[y,...]
    psrc_id = pos_graph.ndata[NID][psrc]
    pdst_id = pos_graph.ndata[NID][pdst]
    raw_pos_edges = torch.zeros((len(psrc_id),3))
    raw_pos_edges[:,0] = psrc_id
    raw_pos_edges[:,1] = pdst_id
    raw_pos_edges[:,2] = torch.tensor([1]*len(raw_pos_edges)) # shape(N,3)

    nsrc, ndst = neg_graph.edges()
    nsrc_id = neg_graph.ndata[NID][nsrc]
    ndst_id = neg_graph.ndata[NID][ndst]
    raw_neg_edges = torch.zeros((len(nsrc_id),3))
    raw_neg_edges[:,0] = nsrc_id
    raw_neg_edges[:,1] = ndst_id
    raw_neg_edges[:,2] = torch.tensor([0]*len(raw_neg_edges))

    raw_edges = torch.cat([raw_pos_edges, raw_neg_edges], dim=0).to(torch.int64)

    new_edge_pairs, labels = undirected_label2directed_label(adj, raw_edges, task, args.negative_ratio) # shape (N,3)
    return new_edge_pairs, labels

def predictor(all_edges, am_outputs, ph_outputs, task, device):
    """
    Perform edge prediction based on input features and task type.

    Args:
    - all_edges (torch.Tensor): Tensor containing all edge pairs in the format (source, destination).
    - am_outputs (torch.Tensor): Tensor containing amplitude outputs.
    - ph_outputs (torch.Tensor): Tensor containing phase outputs.
    - task (int): Specifies the task to perform. 1 for existence prediction, 2 for direction prediction, 3 for three-type classification and 4 for four-type classification.
    - device (torch.device): Device to perform computations on.

    Returns:
    - neg_predicted (torch.Tensor): Negative of the predicted scores for each edge type.
    - labels (torch.Tensor): Predicted labels for the edge pairs.

    Note:
    The predicted labels are as follows:
    - 0: Reverse edges (rev)
    - 1: Positive edges (pos)
    - 2: Bidirectional edges (bi)
    - 3: Non-existent edges (non)
    """

    cos = torch.cos(ph_outputs[all_edges[:,0]] - ph_outputs[all_edges[:,1]]) # N*d
    sin = torch.sin(ph_outputs[all_edges[:,0]] - ph_outputs[all_edges[:,1]])
    mul = am_outputs[all_edges[:,0]].mul(am_outputs[all_edges[:,1]])
    real_part = mul.mul(cos).sum(dim=1)
    img_part = mul.mul(sin).sum(dim=1)

    predicted = torch.zeros(len(all_edges),4).to(device)
    predicted[:,0] = torch.abs(real_part)+torch.abs(img_part+1) # rev
    predicted[:,1] = torch.abs(real_part)+torch.abs(img_part-1) # pos
    predicted[:,2] = torch.abs(real_part-1)+torch.abs(img_part) # bi
    predicted[:,3] = torch.abs(real_part)+torch.abs(img_part) # non
    
    if task == 1:
        labels = torch.argmin(predicted,dim=1)
        # existence prediction 
        labels[labels == 3] = 0 # non
        labels[labels == 2] = 1  # bi
    elif task == 2:
        indices = torch.tensor([0,1]).to(device)
        predicted = torch.index_select(predicted, 1, indices)
        labels = torch.argmin(predicted,dim=1)
    elif task == 3:
        indices = torch.tensor([0,1,3]).to(device)
        predicted = torch.index_select(predicted, 1, indices)
        labels = torch.argmin(predicted,dim=1)
        labels[labels==2] = 3 # predicted labels
    else:
        labels = torch.argmin(predicted,dim=1)
    return -predicted, labels
    
class fourClassLoss(nn.Module):
    """
    Custom loss function for four-class classification tasks.

    Args:
    - args (Namespace): Arguments containing necessary parameters.
    
    Attributes:
    - softmax (nn.Softmax): Softmax layer for probability calculation.
    - celoss (nn.CrossEntropyLoss): Cross entropy loss with class weights.
    - bceloss (nn.BCEWithLogitsLoss): Binary cross entropy loss with logits.

    Methods:
    - forward(all_edges, am_outputs, ph_outputs, loss_weight): Computes the loss based on input data and loss weight.

    Note:
    - idx_1: Reverse edges
    - idx_2: Positive edges
    - idx_3: Bidirectional edges
    - idx_4: Non-existent edges
    """
    def __init__(self, args):
        super(fourClassLoss, self).__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=1)
        self.celoss = nn.CrossEntropyLoss(weight = torch.tensor([1.0,1.0,1.0,1.0/args.negative_ratio]))
        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, all_edges, am_outputs, ph_outputs, loss_weight):
        """
        Compute the loss based on input data and loss weight.

        Args:
        - all_edges (torch.Tensor): Tensor containing all edge pairs in the format (source, destination, label).
        - am_outputs (torch.Tensor): Tensor containing amplitude outputs.
        - ph_outputs (torch.Tensor): Tensor containing phase outputs.
        - loss_weight (float): Weight for the connection-aware loss.

        Returns:
        - loss (torch.Tensor): Total loss computed based on input data and loss weight.
        """
        
        idx_1 = (all_edges[:,2] == 0) #rev
        idx_2 = (all_edges[:,2] == 1) #pos
        idx_3 = (all_edges[:,2] == 2) #bi
        idx_4 = (all_edges[:,2] == 3) #non

        cos = torch.cos(ph_outputs[all_edges[:,0]] - ph_outputs[all_edges[:,1]]) # N*d
        sin = torch.sin(ph_outputs[all_edges[:,0]] - ph_outputs[all_edges[:,1]])
        mul = am_outputs[all_edges[:,0]].mul(am_outputs[all_edges[:,1]])
        real_part = mul.mul(cos).sum(dim=1)
        img_part = mul.mul(sin).sum(dim=1)

        # --------- connection aware loss ------------
        bi_predict = mul.sum(dim=1)
        ex_target = torch.ones(len(all_edges)).to(self.args.device)
        ex_target[idx_4] = 0.0
        exist_loss = self.bceloss(bi_predict, ex_target)
        ## ------------ END --------------
        
        predicted_2 = torch.zeros(len(all_edges),4).to(self.args.device)
        predicted_2[:,0] = -torch.sqrt((real_part)**2+(img_part+1)**2) # rev
        predicted_2[:,1] = -torch.sqrt((real_part)**2+(img_part-1)**2) # pos
        predicted_2[:,2] = -torch.sqrt((real_part-1)**2+(img_part)**2) # bi
        predicted_2[:,3] = -torch.sqrt((real_part)**2+(img_part)**2) # non

        di_target = torch.zeros((len(all_edges), 4)).to(self.args.device)
        di_target[:,0] = idx_1.to(float)
        di_target[:,1] = idx_2.to(float)
        di_target[:,2] = idx_3.to(float)
        di_target[:,3] = idx_4.to(float)
        loss = self.celoss(predicted_2, di_target) + loss_weight*exist_loss
        return loss

class superLoss(nn.Module):
    '''
    supervised loss
    '''
    def __init__(self, args):
        super(superLoss, self).__init__()
        self.args = args
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, pred_score, pred_label):
        loss = self.celoss(pred_score, pred_label)
        return loss