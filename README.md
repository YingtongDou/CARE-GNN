# CARE-GNN

A PyTorch implementation for the [CIKM 2020](https://www.cikm2020.org/) paper below:  
**Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters**.  
[Yingtong Dou](http://ytongdou.com/), [Zhiwei Liu](https://sites.google.com/view/zhiwei-jim), [Li Sun](https://www.researchgate.net/profile/Li_Sun118), Yutong Deng, [Hao Peng](https://penghao-buaa.github.io/), [Philip S. Yu](https://www.cs.uic.edu/PSYu/).  
\[[Paper](https://arxiv.org/pdf/2008.08692.pdf)\]\[[Toolbox](https://github.com/safe-graph/DGFraud)\]\[[DGL Example](https://github.com/dmlc/dgl/tree/master/examples/pytorch/caregnn)\]\[[Benchmark](https://paperswithcode.com/paper/enhancing-graph-neural-network-based-fraud)\]

## Bug Fixes and Update (06/2021)

### Similarity score
The feature and label similarity scores presented in Table 2 of the paper are incorrect. The updated equations for calculating two similarity scores are shown below:
<p align="center">
    <br>
    <a href="https://github.com/YingtongDou/CARE-GNN">
        <img src="https://github.com/YingtongDou/CARE-GNN/blob/master/eq_simi.png" width="500"/>
    </a>
    <br>
<p>

The code for calculating the similarity scores is in [simi_comp.py](https://github.com/YingtongDou/CARE-GNN/blob/master/simi_comp.py).

The updated similarity scores for the two datasets are shown below. Note that we only compute the similarity scores for positive nodes to demonstrate the camouflage of fraudsters (positive nodes).

| YelpChi  | rur  | rtr  | rsr  | homo  |
|-------|--------|--------|--------|--------|
| Avg. Feature Similarity | 0.991   |   0.988    |  0.988  | 0.988  |
| Avg. Label Similarity |  0.909  |   0.176   |  0.186  | 0.184  |

| Amazon  | upu  | usu  | uvu  | homo  |
|-------|--------|--------|--------|--------|
| Avg. Feature Similarity | 0.711   |   0.687    |  0.697  | 0.687  |
| Avg. Label Similarity |  0.167  |   0.056   |  0.053  | 0.072  |

### Relation weight in Figure 3

According to this [issue](https://github.com/YingtongDou/CARE-GNN/issues/5), the weighted aggregation of CARE-Weight (a variant of CARE-GNN) has an error. After fixing it, the relation weight will not converge to the same value. Thus, the relation weight subfigure in Figure 3 and its associated conclusion are wrong.

### Extended version CARE-GNN

Please check out [RioGNN](https://github.com/safe-graph/RioGNN), a GNN model extended based on CARE-GNN with more reinforcement learning modules integrated. We are actively developing an efficient multi-layer version of CARE-GNN. Stay tuned.


## Overview

<p align="center">
    <br>
    <a href="https://github.com/YingtongDou/CARE-GNN">
        <img src="https://github.com/YingtongDou/CARE-GNN/blob/master/model.png" width="900"/>
    </a>
    <br>
<p>

**CA**mouflage-**RE**sistant **G**raph **N**eural **N**etwork **(CARE-GNN)** is a GNN-based fraud detector based on a multi-relation graph equipped with three modules that enhance its performance against camouflaged fraudsters.

Three enhancement modules are:

- **A label-aware similarity measure** which measures the similarity scores between a center node and its neighboring nodes;
- **A similarity-aware neighbor selector** which leverages top-p sampling and reinforcement learning to select the optimal amount of neighbors under each relation;
- **A relation-aware neighbor aggregator** which directly aggregates information from different relations using the optimal neighbor selection thresholds as weights.

CARE-GNN has following advantages:

- **Adaptability.** CARE-GNN adaptively selects best neighbors
for aggregation given arbitrary multi-relation graph;
- **High-efficiency.** CARE-GNN has a high computational efficiency without attention and deep reinforcement learning;
- **Flexibility.** Many other neural modules and external knowledge can be plugged into the CARE-GNN;

We have integrated more than **eight** GNN-based fraud detectors as a TensorFlow [toolbox](https://github.com/safe-graph/DGFraud). 

## Setup

You can download the project and install the required packages using the following commands:

```bash
git clone https://github.com/YingtongDou/CARE-GNN.git
cd CARE-GNN
pip3 install -r requirements.txt
```

To run the code, you need to have at least **Python 3.6** or later versions. 

## Running

1. In CARE-GNN directory, run `unzip /data/Amazon.zip` and `unzip /data/YelpChi.zip` to unzip the datasets; 
2. Run `python data_process.py` to generate adjacency lists used by CARE-GNN;
3. Run `python train.py` to run CARE-GNN with default settings.

For other dataset and parameter settings, please refer to the arg parser in `train.py`. Our model supports both CPU and GPU mode.

## Running on your datasets

To run CARE-GNN on your datasets, you need to prepare the following data:

- Multiple-single relation graphs with the same nodes where each graph is stored in `scipy.sparse` matrix format, you can use `sparse_to_adjlist()` in `utils.py` to transfer the sparse matrix into adjacency lists used by CARE-GNN;
- A numpy array with node labels. Currently, CARE-GNN only supports binary classification;
- A node feature matrix stored in `scipy.sparse` matrix format. 

### Repo Structure
The repository is organized as follows:
- `data/`: dataset files;
- `data_process.py`: transfer sparse matrix to adjacency lists;
- `graphsage.py`: model code for vanilla [GraphSAGE](https://github.com/williamleif/graphsage-simple/) model;
- `layers.py`: CARE-GNN layers implementations;
- `model.py`: CARE-GNN model implementations;
- `train.py`: training and testing all models;
- `utils.py`: utility functions for data i/o and model evaluation.

## Citation
If you use our code, please cite the paper below:
```bibtex
@inproceedings{dou2020enhancing,
  title={Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters},
  author={Dou, Yingtong and Liu, Zhiwei and Sun, Li and Deng, Yutong and Peng, Hao and Yu, Philip S},
  booktitle={Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM'20)},
  year={2020}
}
```
