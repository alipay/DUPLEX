import os
import sys
sys.path.append('../')
from duplex.model import DUPLEX_gat
import tqdm
from config import const_args
import torch
import numpy as np
import pdb
import argparse
import time
from evaluation import evaluation, acc
from data_preprocessing import load_training_data, load_testing_data
from duplex.utils import predictor, fourClassLoss, edge_sampler
from dgl.base import NID, EID
from duplex.mylogging import getLogger
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

import random
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
parser.add_argument('--edge',type=str, default='')
parser.add_argument('--head',type=int, default=1)
args = parser.parse_args()

saveing_args = {
    'save_model_path':"./train_edge/save_models/%s/%s/%i/%i/"%(args.use_model, args.dataset,args.seed,args.loss_weight),
    'save_result_path':"./train_edge/save_models/%s/%s/%i/%i/"%(args.use_model, args.dataset,args.seed,args.loss_weight),
    'training_path':'../edge_data/%s/'%(args.dataset),
    'initial_path':'../edge_data/%s/'%(args.dataset),
    'log_path':"./train_edge/logs/%s/%s/%i/%i/"%(args.use_model, args.dataset,args.seed,args.loss_weight),
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
    logger.info('successfully build log path: %s'%str(os.listdir('./train_edge/')))
    logger.info(args)
    if args.use_model == 'DUPLEX_gat':
        model = DUPLEX_gat(args).to(args.device)
    myloss = fourClassLoss(args).to(args.device)
    loss_weight = args.loss_weight
    early_stop = 0 # early-stopping counter
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # training data
    train_dataloader, whole_graph, embedding_matrix = load_training_data(args)
    whole_graph = whole_graph.to(args.device)
    embedding_matrix = embedding_matrix.to(args.device)
    print(train_dataloader.device)          
    logger.info('successfully loaded training data, size%i'%whole_graph.num_nodes())
    
    # testing and validation data
    test_blocks = {}
    val_blocks = {}
    test_edges = {}
    val_edges = {}
    for task in eval_task:
        test_blocks[task], test_edges[task],val_blocks[task],val_edges[task] = load_testing_data(args, task, whole_graph)
    logger.info('successfully loaded testing data')
    
    # cross validation
    best_accuracy = [0,0,0,0,0]
    best_model_path = args.save_model_path 
    best_result_path = args.save_result_path
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(best_result_path):
        os.makedirs(best_result_path)

    am_embedding = embedding_matrix[:,:args.input_dim].to(args.device)
    ph_embedding = embedding_matrix[:,args.input_dim:].to(args.device)
    
    # --------------- training ---------------------
    for epoch in range(args.epochs): 
        for step, (input_nodes, pos_graph, blocks) in enumerate(train_dataloader): 
            input_am, input_ph = blocks[0].srcdata['h'][:,:args.input_dim], blocks[0].srcdata['h'][:,args.input_dim:]
            output_nodes = blocks[-1].dstdata[NID] # initial ids of output nodes
            am_outputs, ph_outputs = model(blocks, input_am, input_ph) # am: n*d, ph: n*1 # nodes in a batch
            pos_edges = torch.cat((pos_graph.edges()[0].unsqueeze(-1),pos_graph.edges()[1].unsqueeze(-1),pos_graph.edata['label'].unsqueeze(-1)),1)
            pos_edges[pos_edges[:,2]==0] = torch.index_select(pos_edges[pos_edges[:,2]==0], 1, torch.tensor([1,0,2]).to(args.device))
            batch_edges = pos_edges
            loss = myloss(batch_edges, am_outputs, ph_outputs, loss_weight)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_label = batch_edges[:,2]
            train_score, train_pred_label = predictor(batch_edges[:,:2], am_outputs.detach(), ph_outputs.detach(), 4, args.device) # shape (N,3)
            train_score, train_pred_label = train_score, train_pred_label
            train_acc= acc(train_pred_label, train_label,4) # four-type acc
            am_embedding[output_nodes,:] = am_outputs.detach()
            ph_embedding[output_nodes,:] = ph_outputs.detach()
            
        # ------ validation --------
        valid_acc = [0,0,0,0,0]
        valid_auc = [0,0,0,0,0]
        if (epoch)%10 == 0:
            model.eval()
            for task in eval_task:
                input_am_val, input_ph_val = val_blocks[task][0].srcdata['h'][:,:args.input_dim], val_blocks[task][0].srcdata['h'][:,args.input_dim:]
                output_nodes_val = val_blocks[task][-1].dstdata[NID]
                am_outputs_val, ph_outputs_val = model(val_blocks[task], input_am_val, input_ph_val)
                
                val_label = val_edges[task][:,2]
                global2batch = torch.zeros(max(output_nodes_val)+1,dtype=int).to(args.device)
                global2batch[output_nodes_val]=torch.range(0, am_outputs_val.shape[0]-1,dtype=int).to(args.device) # global idx to batch idx
                batch_edges = torch.cat((global2batch[val_edges[task][:,0]].unsqueeze(1),global2batch[val_edges[task][:,1]].unsqueeze(1),val_label.unsqueeze(1)),dim=1)

                val_score, pred_val = predictor(batch_edges[:,:2], am_outputs_val, ph_outputs_val, task, args.device) # shape (N,3)
                valid_acc[task] = acc(pred_val, val_label, task)

                if best_accuracy[task] < valid_acc[task]:
                    early_stop = 0
                    best_accuracy[task] = valid_acc[task]
                    torch.save(model.state_dict(), best_model_path+'model%s_%s.pt'%(args.save_log, task))
                    np.savez(best_result_path+'result_val%s_%s.npz'%(args.save_log, task), am_embedding = am_embedding.cpu().numpy(), ph_embedding = ph_embedding.cpu().numpy(), 
                        test_edge = val_edges[task][:,:2].cpu().numpy(), pred_label = pred_val.cpu().numpy(), test_label = val_label.cpu().numpy())

                else:
                    early_stop += 1
        
        model.train()
        
        # ---------- print information -------
        if (epoch)%400 == 0:
            logger.info("Epoch {} training loss {} acc {}".format(epoch, loss, train_acc))
            logger.info("Epoch {} Validation ACC {}".format(epoch, valid_acc[1:]))

        if (epoch>1000) & (early_stop > args.early_stop):
            print('EARLY STOP!')
            break

        loss_weight = loss_weight*(1-args.loss_decay) # connection-aware loss weight decay

    # ------- testing ---------
    model.eval()
    for task in eval_task:
        state_dict = torch.load(best_model_path+'model%s_%s.pt'%(args.save_log, task))
        model.load_state_dict(state_dict)

        input_am_test, input_ph_test = test_blocks[task][0].srcdata['h'][:,:args.input_dim], test_blocks[task][0].srcdata['h'][:,args.input_dim:]
        output_nodes_test = test_blocks[task][-1].dstdata[NID]
        am_outputs_test, ph_outputs_test = model(test_blocks[task], input_am_test, input_ph_test) 
        
        am_embedding = torch.zeros(embedding_matrix[:,:args.input_dim].shape)
        ph_embedding = torch.zeros(embedding_matrix[:,:args.input_dim].shape)
        am_embedding[output_nodes_test,:] = am_outputs_test.detach().cpu()
        ph_embedding[output_nodes_test,:] = ph_outputs_test.detach().cpu()
        global2batch = torch.zeros(max(output_nodes_test)+1,dtype=int).to(args.device)
        global2batch[output_nodes_test]=torch.range(0, am_outputs_test.shape[0]-1,dtype=int).to(args.device)
        batch_edges = torch.cat((global2batch[test_edges[task][:,0]].unsqueeze(1),global2batch[test_edges[task][:,1]].unsqueeze(1)),dim=1)

        test_score, pred_label = predictor(batch_edges, am_outputs_test, ph_outputs_test, task, args.device) # shape (N,3)
        test_score, pred_label = test_score.detach().cpu(), pred_label.detach().cpu()
        test_label = test_edges[task][:,2].cpu()
        test_acc, test_auc, _ = evaluation(task, test_score, pred_label, test_label)
        np.savez(best_result_path+'best_result%s_%s.npz'%(args.save_log, task), am_embedding = am_embedding.cpu().detach().numpy(), ph_embedding = ph_embedding.cpu().detach().numpy(), 
            test_edge = test_edges[task][:,:2].cpu().numpy(), pred_score = test_score.numpy(), pred_label = pred_label.numpy(), test_label = test_label.numpy())
        logger.info("Task {}, Test ACC {} AUC {}".format(task, test_acc, test_auc))

if __name__ == '__main__':
    try:
        eval_task = [1,2,3,4]
        main()
    except KeyboardInterrupt:
        np.savetxt(args.save_result_path+'embedding%s.txt'%(args.save_log), embedding_matrix.cpu().numpy(), fmt='%f', delimiter=',')
        sys.exit()