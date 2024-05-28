import os
from itertools import product
use_model = 'DUPLEX_gat' # model
lrs = [1e-3] # learning rate
dataset = 'cora_ml'
seeds = range(10)
fusions = ['add'] # fusion type: without fusion or with fusion
log_note = 'node'
bc_size = 2048*3 # batch size
n_layers = 2 # network layers
feature = 'init'
task = 'inductive'
fusion_layer = 1
heads = [1] # attention heads

if dataset == 'citeseer':
    labels = 6
    input_dim = 3703

if dataset == 'cora_ml':
    labels = 7
    input_dim = 2879

for lr in lrs:
    for fusion in fusions:
        for seed in seeds:
            for head in heads:
                save_log="seed%s_layer%s_fusion%s_use_model%s_task%s"%(seed, n_layers, fusion, use_model, task)
                order = "python ./train_node_ind/training.py --m %s \
                        --input_dim %s \
                        --learning_rate %s\
                        --dataset %s \
                        --labels %s \
                        --seed %s \
                        --fusion %s \
                        --save_log %s\
                        --bc_size %s\
                        --n_layers %s\
                        --use_model %s\
                        --feature %s\
                        --task %s\
                        --fusion_layer %s\
                        --head %s"%(save_log, input_dim, lr, \
                            dataset, labels, seed, fusion, save_log, \
                                bc_size, n_layers, use_model, feature, task, fusion_layer,head)
                os.system(order)
