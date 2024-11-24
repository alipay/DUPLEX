# DUPLEX: Dual Graph Attention Network for Complex Embedding of Directed Graphs

This PyTorch implementation showcases the DUPLEX model as described in the paper [DUPLEX: Dual GAT for Complex Embedding of Directed Graphs].

![The network architecture of DUPLEX](./duplex.jpg)
<center><b>Figure 1:</b> The network architecture of DUPLEX.</center>

## Requirements
- Ubuntu OS
- Python 3.8
- PyTorch 2.0.1
- CUDA 11.4

You can install the necessary dependencies using the following command:
```bash
conda env create -f environment.yml
```

## Data Preparation

To preprocess data, follow these steps:

1. Navigate to the `./code/` directory using the command `cd ./code/`.
2. Open the `./generate_data.ipynb` notebook to generate DGL graphs from raw data.
3. Execute `python ./train_edge/split_data.py` to create the train/validation/test sets for the link prediction task.
4. Execute `python ./train_node_ind/process_data.py` to create the train/validation/test sets for the node classification task.

## Training & Evaluation

To conduct experiments, follow these steps:

1. Navigate to the `./code/` directory using the command `cd ./code/`.
2. Execute `python ./train_edge/train.py` for link prediction.
3. Execute `python ./train_node_trans/train.py` for transductive node classification.
4. Execute `python ./train_node_ind/train.py` for inductive node classification.

For details on each command-line argument, please refer to the explanations provided in the respective training scripts.

## Citation

```@InProceedings{pmlr-v235-ke24c,
  title = 	 {{DUPLEX}: Dual {GAT} for Complex Embedding of Directed Graphs}, 
  author =       {Ke, Zhaoru and Yu, Hang and Li, Jianguo and Zhang, Haipeng},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {23430--23448},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/ke24c/ke24c.pdf},
  url = 	 {https://proceedings.mlr.press/v235/ke24c.html},
  abstract = 	 {Current directed graph embedding methods build upon undirected techniques but often inadequately capture directed edge information, leading to challenges such as: (1) Suboptimal representations for nodes with low in/out-degrees, due to the insufficient neighbor interactions; (2) Limited inductive ability for representing new nodes post-training; (3) Narrow generalizability, as training is overly coupled with specific tasks. In response, we propose DUPLEX, an inductive framework for complex embeddings of directed graphs. It (1) leverages Hermitian adjacency matrix decomposition for comprehensive neighbor integration, (2) employs a dual GAT encoder for directional neighbor modeling, and (3) features two parameter-free decoders to decouple training from particular tasks. DUPLEX outperforms state-of-the-art models, especially for nodes with sparse connectivity, and demonstrates robust inductive capability and adaptability across various tasks. The code will be available upon publication.}
}
```

## Contact

For any questions related to DUPLEX, please submit them to Github Issues.