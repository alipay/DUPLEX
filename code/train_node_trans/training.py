import os
import sys
sys.path.append('../')
from duplex.model import DUPLEX_gat
from data_preprocessing import load_graph_data, read_initialize
import numpy as np
from dgl.base import NID, EID

import tqdm
from config import const_args
import torch
import pdb
import argparse
import time
from torch.profiler import profile, record_function, ProfilerActivity

from data_preprocessing import load_graph_data, load_train_test_data
from duplex.utils import fourClassLoss
from evaluation import node_eval
from dgl.base import NID, EID
from duplex.mylogging import getLogger
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import random
import pickle

# save
def save_pkl(file, model):
    with open(file,'wb') as f:
        pickle.dump(model,f)

# load
def load_pkl(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
# ------------ argument ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--m', type=str, default=None, dest = 'note', help='anything wants to note')
parser.add_argument('--input_dim', type=int, default=128)
parser.add_argument('--loss_weight', type=float, default=0.1, dest = 'loss_weight', help='bce loss weight')
parser.add_argument('--negative_ratio', type=int, default=1, dest = 'negative_ratio', help='negative edges when training')
parser.add_argument('--dataset', type=str, default='citeseer', dest = 'dataset', help='dataset name')
parser.add_argument('--seed', type=int, default=0, dest = 'seed', help='data seed')
parser.add_argument('--save_log', type=str, default='', dest = 'save_log', help='True if save log and model, with name')
parser.add_argument('--loss_decay', type=float, default=0, dest = 'loss_decay', help='weight decay per epoch')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--fusion', type=str, default=None)
parser.add_argument('--bc_size',type=int, default=4096)
parser.add_argument('--n_layers',type=int, default=3)
parser.add_argument('--use_model',type=str, default='DUPLEX')
parser.add_argument('--head',type=int, default=1)
parser.add_argument('--fusion_layer',type=int, default=1)
args = parser.parse_args()


saveing_args = {
    'save_model_path':"./train_node_trans/save_models/%s/%s/%i/%i/"%(args.use_model, args.dataset,args.seed,args.loss_weight),
    'save_result_path':"./train_node_trans/save_models/%s/%s/%i/%i/"%(args.use_model, args.dataset,args.seed,args.loss_weight),
    'training_path':'../node_data/%s/'%(args.dataset),
    'initial_path':'../node_data/%s/'%(args.dataset),
    'log_path':"./train_node_trans/logs/%s/%s/%i/%i/"%(args.use_model, args.dataset,args.seed,args.loss_weight),
}

args_dict = vars(args)  
args_dict.update(const_args)
args_dict.update(saveing_args)
args = argparse.Namespace(**args_dict)

set_random_seed(args.seed)

# ------------- training & evaluation ---------------

def main():
    print(args)
    logger = getLogger(args)
    logger.info('successfully build log path: %s'%str(os.listdir('../')))
    logger.info(args)
    if args.use_model == 'DUPLEX_gat':
        model = DUPLEX_gat(args).to(args.device)
        
    myloss = fourClassLoss(args).to(args.device)
    loss_weight = args.loss_weight

    early_stop = 0
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # training data
    train_dataloader, whole_graph, feature_matrix, all_dataloader = load_graph_data(args)
    print(train_dataloader.device)          
    logger.info('successfully loaded graph data, size %i, dim (%i,%i)'%(whole_graph.num_nodes(), feature_matrix.shape[0], feature_matrix.shape[1]))

    # cross validation    
    best_accuracy = 0 # cross validation
    best_f1 = 0
    best_loss = 10000
    best_model_path = args.save_model_path 
    best_result_path = args.save_result_path
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(best_result_path):
        os.makedirs(best_result_path)

    embedding_matrix = torch.zeros((whole_graph.num_nodes(), args.output_dim*2))
    # --------------- training ---------------------
    for epoch in range(args.epochs):
        for step, (input_nodes, pos_graph, blocks) in enumerate(train_dataloader):
            input_am, input_ph = blocks[0].srcdata['h'][:,:args.input_dim], blocks[0].srcdata['h'][:,args.input_dim:]
            output_nodes = blocks[-1].dstdata[NID] # initial ids of output nodes
            am_outputs, ph_outputs = model(blocks, input_am, input_ph)  # am: n*d, ph: n*1 # nodes in a batch
            
            pos_edges = torch.cat((pos_graph.edges()[0].unsqueeze(-1),pos_graph.edges()[1].unsqueeze(-1),pos_graph.edata['label'].unsqueeze(-1)),1)
            pos_edges[pos_edges[:,2]==0] = torch.index_select(pos_edges[pos_edges[:,2]==0], 1, torch.tensor([1,0,2]).to(args.device))
            batch_edges = pos_edges
            loss = myloss(batch_edges, am_outputs, ph_outputs, loss_weight)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
        # ------ validation --------
        # fit on training data
        if (epoch > 800) & (epoch % 10 == 0):
            if loss < best_loss:
                early_stop = 0
                torch.save(model.state_dict(), best_model_path+'model_%s.pt'%(args.save_log))
            else:
                early_stop += 1
                
        # ---------- print information -------
        if (epoch)%400 == 0:
            logger.info("Epoch {} training loss {}".format(epoch, loss))

        if (epoch>800) & (early_stop > args.early_stop):
            print('EARLY STOP!')
            break

        loss_weight = loss_weight*(1-args.loss_decay)

    # ---- testing ------
    logger.info('testing start')
    for step, (input_nodes, output_nodes, blocks) in enumerate(all_dataloader): 
        input_am, input_ph = blocks[0].srcdata['h'][:,:args.input_dim], blocks[0].srcdata['h'][:,args.input_dim:]
        output_nodes = blocks[-1].dstdata[NID]
        am_outputs, ph_outputs = model(blocks, input_am, input_ph)
        embedding_matrix[output_nodes.cpu(),:args.output_dim] = am_outputs.cpu().detach()
        embedding_matrix[output_nodes.cpu(),args.output_dim:] = ph_outputs.cpu().detach()
    
    np.savetxt(best_result_path+'embeddings_%s.txt'%(args.save_log), embedding_matrix)
    
    for seed in range(10):
        args.seed = seed

        train_samples, val_samples, _, test_samples, _ = load_train_test_data(args, whole_graph)
        logger.info('successfully loaded testing data')
        embedding_matrix = torch.tensor(np.loadtxt(best_result_path+'embeddings_%s.txt'%(args.save_log)).astype(np.float32))
        labels = torch.tensor(np.loadtxt(args.initial_path+'labels.txt').astype(int))
        logger.info('embedding shape: %s, %s'%embedding_matrix.shape)
        
        initial_f = []
        test_pred, test_f1_ma, test_f1_mi = node_eval(train_samples, val_samples, test_samples, embedding_matrix, labels, args.device)
        np.savez(best_result_path+'best_result_%s.npz'%(args.save_log), embedding_matrix = embedding_matrix, 
                        test_node= test_samples, pred_label = test_pred)
        logger.info("Test macro f1 {} micro f1 {}".format(test_f1_ma, test_f1_mi))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        np.savetxt(args.save_result_path+'embedding_%s.txt'%(args.save_log), embedding_matrix.numpy(), fmt='%f', delimiter=',')
        sys.exit()