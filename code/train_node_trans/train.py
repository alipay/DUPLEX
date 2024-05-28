import os
from itertools import product
use_model = 'DUPLEX_gat' # model
loss_weights = [0.1,0.3] # loss weight λ
loss_decays = [0,1e-4,1e-2] # decay factor q
lrs = [1e-3] # learning rate
weight_decays = [0]
dataset = 'citeseer'
seeds = range(10)
fusions = ['none','add'] # fusion type: without fusion or with fusion
log_note = 'node'
bc_size = 2048*4 # batch size
n_layers = 3 # network layers
if_concat = ''
fuse_layer = 1
heads = [1] # attention heads

if dataset == 'citeseer':
    labels = 6
    input_dim = 128
if dataset == 'cora_ml':
    labels = 7
    input_dim = 128

for (lw, ld, lr, wd, head) in product(loss_weights,loss_decays,lrs,weight_decays, heads):
    for fusion in fusions:
        save_log="no_sup_head%s_lw%s_ld%s_layer%s_fusion%s_use_model%s_%s_%s"%(head, lw, ld, n_layers, fusion, use_model, log_note, if_concat)
        order = "python ./train_node_trans/training%s.py --m %s \
                --input_dim %s \
                --loss_weight %s \
                --loss_decay %s\
                --learning_rate %s\
                --dataset %s \
                --fusion %s \
                --save_log %s\
                --bc_size %s\
                --n_layers %s\
                --use_model %s\
                --fusion_layer %s\
                --head %s"%(if_concat, save_log, input_dim, lw, ld, lr, dataset, fusion, save_log, bc_size, n_layers, use_model, fuse_layer, head)
        os.system(order)
