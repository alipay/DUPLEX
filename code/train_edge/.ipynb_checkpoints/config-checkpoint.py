import torch
import argparse
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    num_workers = 0
    use_uva = False
else:
    num_workers = 0
    use_uva = True

const_args = {
    'device':device,
    'input_dim':128,
    'hidden_dim':128,
    'output_dim':128,
    'bias':True,
    'dr_rate':0.5, # dropout rate
    'negative_weight':1, # negative sampling weight
    'epochs':3000,
    'early_stop':100,
    'num_workers':num_workers,
    'use_uva':use_uva,
    'test_val_ratio':[0.15,0.05],
    }