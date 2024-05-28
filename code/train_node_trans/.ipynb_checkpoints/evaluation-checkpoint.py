import torch.nn as nn
import torch
import numpy as np
from collections.abc import Mapping
import pdb
import pandas
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, f1_score

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, input_dim*2))
        self.linears.append(nn.Linear(input_dim*2, output_dim))
        self.activation = nn.functional.relu
        self.dropout = nn.Dropout(0.3)
    def forward(self, features):
        f = features
        for i in range(len(self.linears)):
            f = self.linears[i](f)
            if i < len(self.linears)-1:
                f = self.activation(f)
                f = self.dropout(f)
        return f

def cal_f1(score, label):
    """
    Calculate F1 scores given prediction scores and true labels.

    Args:
    - score (torch.Tensor): Prediction scores.
    - label (torch.Tensor): True labels.

    Returns:
    - pred (torch.Tensor): Predicted labels.
    - ma_f1 (float): Macro F1 score.
    - mi_f1 (float): Micro F1 score.
    """
    score = score.detach()
    label = label.cpu()
    pred = score.argmax(dim=-1).cpu()
    ma_f1 = f1_score(label, pred, average='macro')
    mi_f1 = f1_score(label, pred, average='micro')
    return pred, ma_f1, mi_f1

def node_eval(train_samples, val_samples, test_samples, features, labels, device):
    """
    Evaluate node classification using a Multi-Layer Perceptron (MLP) model.

    Args:
    - train_samples (torch.Tensor): Training samples (node indices and labels).
    - val_samples (torch.Tensor): Validation samples (node indices and labels).
    - test_samples (torch.Tensor): Test samples (node indices and labels).
    - features (torch.Tensor): Node features.
    - labels (torch.Tensor): Unique node labels.
    - device (str): Device to run the evaluation on (e.g., 'cpu', 'cuda').

    Returns:
    - pred_label_test (torch.Tensor): Predicted labels for the test set.
    - mac_f1_test (float): Macro F1 score on the test set.
    - mic_f1_test (float): Micro F1 score on the test set.
    """

    # Determine the number of unique labels
    labels = len(labels.unique())
    
    # Move data to the specified device
    train_samples = train_samples.to(device)
    val_samples = val_samples.to(device)
    test_samples = test_samples.to(device)
    
    # Initialize CrossEntropyLoss and MLP model
    celoss = nn.CrossEntropyLoss().to(device)
    new_feature = features.to(device)
    dims = new_feature.shape[1]
    print('classification feature size: ',new_feature.shape)
    model = MLP(dims, labels).to(train_samples.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Early stopping and best F1 score tracking
    early_stop = 0
    best_f1 = 0
    print('classification device',device)
    
    # Training loop
    for epoch in range(1000):
        pred_score_train = model(new_feature[train_samples[:,0],:])
        _, mac_f1_train, mic_f1_train = cal_f1(pred_score_train, train_samples[:,1])
        loss = celoss(pred_score_train, train_samples[:,1])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            model.eval()
            pred_score_val= model(new_feature[val_samples[:,0],:])
            _, mac_f1_val, mic_f1_val = cal_f1(pred_score_val, val_samples[:,1])
            if mac_f1_val > best_f1:
                early_stop = 0
                best_f1 = mac_f1_val
                torch.save(model.state_dict(), './best_model.pt')
            else:
                early_stop += 1
            model.train()
        if early_stop > 50:
            print('EARLY STOP!!!')
            break
        if (epoch)%400 == 0:
            print("Epoch {} training loss {} macro f1 {} micro f1 {}".format(epoch, loss, mac_f1_train, mic_f1_train))
            print("Epoch {} best validation mac f1 {}, latest val mac f1 {}, mic f1 {}".format(epoch, best_f1, mac_f1_val, mic_f1_val))
    
    # Load the best model and evaluate on the test set
    model.eval()
    state_dict = torch.load('./best_model.pt')
    model.load_state_dict(state_dict)
    print('test: ',str(test_samples[:,1].unique(return_counts=True)))
    pred_score_test = model(new_feature[test_samples[:,0],:])
    pred_label_test, mac_f1_test, mic_f1_test = cal_f1(pred_score_test.cpu(), test_samples[:,1].cpu())
    return pred_label_test, mac_f1_test, mic_f1_test