import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import argparse
import random
import networkx
import pdb
from duplex.utils import predictor
from data_preprocessing import read_initialize
import torch.nn.functional as F

def load_result_npz(npz_file):
    '''
    load .npz files
    '''
    a = np.load(npz_file)
    am_embedding = a['am_embedding']
    ph_embedding = a['ph_embedding']
    test_edge = a['test_edge']
    pred_label = a['pred_label']
    test_label = a['test_label']

    return am_embedding, ph_embedding, test_edge, pred_label, test_label

def load_edges(edges_file):
    '''
    load edges
    '''
    return np.loadtxt(edges_file, dtype=int, delimiter=',')

def load_emb(emb_file, dim):
    '''
    load embeddings
    '''
    embedding_matrix = read_initialize(emb_file)
    am_embedding = embedding_matrix[:,:dim]
    ph_embedding = embedding_matrix[:,dim:]
    return am_embedding, ph_embedding

def evaluation(task, pred_score, pred_label, test_label):
    '''
    Evaluate the classification performance.

    Parameters:
    - task: Integer indicating the type of classification task (1 or 2).
    - pred_score: Predicted scores from the classifier.
    - pred_label: Predicted labels from the classifier.
    - test_label: True labels from the test set.

    Returns:
    - acc: Accuracy of the classifier.
    - auc: Area under the ROC curve (AUC) if applicable (0 if task is not 1 or 2).
    - summary: Classification report containing precision, recall, F1-score, and support.
    '''
    auc = 0
    acc = accuracy_score(test_label, pred_label)
    if task == 1:
        pred_score = F.softmax(pred_score,dim=1)
        pd_score = torch.zeros(pred_score.shape[0],2)
        pd_score[:,0] = pred_score[:,0]+pred_score[:,3]
        pd_score[:,1] = pred_score[:,1]+pred_score[:,2]
        pred_score = pd_score[:,1]
        auc = roc_auc_score(test_label, pred_score)
    if task == 2:
        pred_score = pred_score[:,:2]
        auc = roc_auc_score(test_label, F.softmax(pred_score,dim=1)[:,1])
    summary = classification_report(test_label, pred_label)
    return acc, auc, summary

def acc(pred, label, task):
    correct = pred.eq(label).sum().item()
    acc = correct / len(pred)
    return acc

def main():
    eval_parser = argparse.ArgumentParser()
    eval_parser.add_argument('--edges', type=str,  help='A file containing the testing edges')
    eval_parser.add_argument('--emb', type=str,  help='A file containing the embeddings')
    eval_parser.add_argument('--npz', type=str, help='A file containing the result npz')
    eval_parser.add_argument('--dim', type=int, help='embedding dim size')
    eval_parser.add_argument('--task', type=int, help='link prediction task')
    eval_args = eval_parser.parse_args()

    if eval_args.npz:
        am_embedding, ph_embedding, test_edge, _, test_label = load_result_npz(eval_args.npz)
    if eval_args.edges:
        test_edge = load_edges(eval_args.edges)
        test_label = test_edge[:,2]
    if eval_args.emb:
        am_embedding, ph_embedding = load_emb(eval_args.emb, eval_args.dim)
    pred_score, pred_label = predictor(torch.tensor(test_edge[:,:2]), torch.tensor(am_embedding), torch.tensor(ph_embedding), eval_args.task, 'cpu')

    acc, auc, summary = evaluation(eval_args.task, pred_score, pred_label, test_label)

    print('ACC: {} \nAUC: {}'.format(acc, auc))

if __name__ == '__main__':
    main()
