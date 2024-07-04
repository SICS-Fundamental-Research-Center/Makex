#!/bin/bash
conda create -y -n DAG python=3.8
source activate DAG
conda install -y pytorch==1.8.0 -c pytorch
CUDA="cu102"
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-geometric
pip install tqdm
pip install networkx==2.5