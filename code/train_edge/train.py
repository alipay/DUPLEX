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
if dataset == 'citeseer':
    bc_size = 2048*2 # batch size
else:
    bc_size = 2048*4
n_layers = 3 # network layers
heads = [1] # attention heads
for (lw, ld, lr, wd, head) in product(loss_weights,loss_decays,lrs,weight_decays,heads):
    for fusion in fusions:
        for seed in seeds:
            save_log="head%s_lw%s_ld%s_seed%s_layer%s_fusion%s_use_model%s_l1"%(head, lw, ld, seed, n_layers, fusion, use_model)
            order = "python ./train_edge/training.py --m %s \
                    --loss_weight %s \
                    --loss_decay %s\
                    --learning_rate %s\
                    --dataset %s \
                    --seed %s \
                    --fusion %s \
                    --save_log %s\
                    --bc_size %s\
                    --n_layers %s\
                    --use_model %s\
                    --head %s"%(save_log, lw, ld, lr, dataset, seed, fusion, save_log, bc_size, n_layers, use_model, head)
            os.system(order)
