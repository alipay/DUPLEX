import os
import sys
sys.path.append('../')
from duplex.model import DUPLEX_gat_in
import numpy as np
from dgl.base import NID, EID

import tqdm
from config import const_args
import torch
import pdb
import argparse
import time
from duplex.train_node_trans.evaluation import cal_f1

from data_preprocessing import load_graph_data_transductive, load_train_test_data, load_graph_data_inductive
from duplex.utils import superLoss
from dgl.base import NID, EID
from duplex.mylogging import getLogger
import warnings
warnings.filterwarnings('ignore')
import random
import pickle
import torch.nn as nn

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

# --------- parser ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--m', type=str, default=None, dest = 'note', help='anything wants to note')
parser.add_argument('--input_dim', type=int, default=128)
parser.add_argument('--negative_ratio', type=int, default=1, dest = 'negative_ratio', help='negative edges when training')
parser.add_argument('--dataset', type=str, default='citeseer', dest = 'dataset', help='dataset name')
parser.add_argument('--labels', type=int, default=None)
parser.add_argument('--seed', type=int, default=0, dest = 'seed', help='data seed')
parser.add_argument('--save_log', type=str, default='', dest = 'save_log', help='True if save log and model, with name')
parser.add_argument('--loss_decay', type=float, default=0, dest = 'loss_decay', help='weight decay per epoch')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--fusion', type=str, default=None)
parser.add_argument('--bc_size',type=int, default=4096)
parser.add_argument('--n_layers',type=int, default=3)
parser.add_argument('--use_model',type=str, default='DUPLEX')
parser.add_argument('--feature',type=str, default='init')
parser.add_argument('--balance',type=int, default=0)
parser.add_argument('--task',type=str, default='inductive')
parser.add_argument('--head',type=int, default=1)
parser.add_argument('--fusion_layer',type=int, default=1)
args = parser.parse_args()


saveing_args = {
    'save_model_path':"./train_node_ind/save_models/%s/%s/%i/"%(args.use_model, args.dataset,args.seed),
    'save_result_path':"./train_node_ind/save_models/%s/%s/%i/"%(args.use_model, args.dataset,args.seed),
    'training_path':'../node_data/%s/'%(args.dataset),
    'initial_path':'../node_data/%s/'%(args.dataset),
    'log_path':"./train_node_ind/logs/%s/%s/%i/"%(args.use_model, args.dataset,args.seed),
}

args_dict = vars(args) 
args_dict.update(const_args)
args_dict.update(saveing_args)
args = argparse.Namespace(**args_dict)

set_random_seed(args.seed)

def main():
    print(args)
    logger = getLogger(args)
    logger.info('successfully build log path: %s'%str(os.listdir('../')))
    logger.info(args)
    if args.use_model == 'DUPLEX_gat':
        model = DUPLEX_gat_in(args).to(args.device)

    early_stop = 0
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
    if args.task == 'inductive':
        train_dataloader, whole_graph, feature_matrix, val_blocks = load_graph_data_inductive(args)
        print(train_dataloader.device)          
        logger.info('successfully loaded graph data, size %i, dim (%i,%i)'%(whole_graph.num_nodes(), feature_matrix.shape[0], feature_matrix.shape[1]))
        
        _, _, _, _, test_blocks = load_train_test_data(args, whole_graph)
        logger.info('successfully loaded testing data')
    if args.task == 'transductive':
        train_dataloader, whole_graph, feature_matrix = load_graph_data_transductive(args)
        print(train_dataloader.device)          
        logger.info('successfully loaded graph data, size %i, dim (%i,%i)'%(whole_graph.num_nodes(), feature_matrix.shape[0], feature_matrix.shape[1]))
        
        _, _, val_blocks, _, test_blocks = load_train_test_data(args, whole_graph)
        logger.info('successfully loaded testing data')
    superloss = nn.CrossEntropyLoss().to(args.device)

    best_accuracy = 0
    best_f1 = 0
    best_loss = 10000
    best_model_path = args.save_model_path
    best_result_path = args.save_result_path
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(best_result_path):
        os.makedirs(best_result_path)

    # -------------- whole graph --------------
    for epoch in range(args.epochs): 
        for step, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader): 
            input_am, input_ph = blocks[0].srcdata['h'], blocks[0].srcdata['h']
            output_nodes = blocks[-1].dstdata[NID]
            output_labels = blocks[-1].dstdata['label']
            pred_score = model(blocks, input_am, input_ph)
            loss = superloss(pred_score, output_labels)
            pred_label_train, mac_f1, mic_f1 = cal_f1(pred_score, output_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        # ------ validation --------
        # fit on training data   
        if epoch % 10 ==0:
            model.eval()
            input_am_val, input_ph_val = val_blocks[0].srcdata['h'], val_blocks[0].srcdata['h']
            output_nodes_val = val_blocks[-1].dstdata[NID] 
            output_labels_val = val_blocks[-1].dstdata['label']
            pred_score_val = model(val_blocks, input_am_val, input_ph_val) 
            pred_label_val, mac_f1_val, mic_f1_val = cal_f1(pred_score_val, output_labels_val)
        
            if mac_f1_val > best_f1:
                early_stop = 0
                best_f1 = mac_f1_val
                torch.save(model.state_dict(), best_model_path+'model_%s.pt'%(args.save_log))
            else:
                early_stop += 1
            model.train()
        
        if (epoch)%400 == 0:
            logger.info("Epoch {} training loss {} mac f1 {}, mic f1 {}".format(epoch, loss, mac_f1, mic_f1))
            logger.info("Epoch {} best validation mac f1 {}, latest val mac f1 {}, mic f1 {}".format(epoch, best_f1, mac_f1_val, mic_f1_val))

        if (epoch>1000) & (early_stop > args.early_stop):
            print('EARLY STOP!')
            break

    # ---- testing ------
    model.eval()
    state_dict = torch.load(best_model_path+'model_%s.pt'%(args.save_log))
    model.load_state_dict(state_dict)
    input_am_test, input_ph_test = test_blocks[0].srcdata['h'], test_blocks[0].srcdata['h']
    output_nodes_test = test_blocks[-1].dstdata[NID]
    output_labels_test = test_blocks[-1].dstdata['label']
    pred_score_test = model(test_blocks, input_am_test, input_ph_test)
    pred_label_test, mac_f1_test, mic_f1_test = cal_f1(pred_score_test, output_labels_test)
    logger.info("Test macro f1 {} micro f1 {}".format(mac_f1_test, mic_f1_test))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()