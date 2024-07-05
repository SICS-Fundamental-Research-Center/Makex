# Overview
Makex (MAKE senSE) is a logic approach to explaining why a GNN-based model M(x,y) recommends item y to user x. It proposes a class of Rules for ExPlanations, denoted as REPs and defined with a graph pattern Q and dependency X → M(x,y), where X is a collection of predicates, and the model M(x,y) is treated as the consequence of the rule. Intuitively, given M(x,y), we discover pattern Q to identify relevant topology, and precondition X to disclose correlations, interactions and dependencies of vertex features; together they provide rationals behind prediction M(x,y), identifying what features are decisive for M to make predictions and under what conditions the decision can be made. We (a) define REPs with 1-WL test, on which most GNN models for recommendation are based; (b) develop an algorithm for discovering REPs for M as global explanations, and (c) provide a top-k algorithm to compute top-ranked local explanations.

# Environment Requirement
- python 3.9
- torch 1.8.1
- scikit-learn 1.0.1
  
# install c++ module

```
export CFLAGS='-std=c++17' CC=g++-9 CXX=g++-9
cd pyMakex
./clean.sh
```

# Global Explanations

```
cd global_explanations
./rep_discovery.sh
```


# Local Explanations

```
cd local_explanations
./run_local_explanation.sh
```


# Dataset
We provide the three datasets. All datasets can be downloaded [here](https://drive.google.com/drive/folders/1kyu0PLHg1uRe9LN-c2WlqrcqWkPCu-Yp?usp=sharing).
## Movielens
### movielens_v.csv
- item's label_id is 0;
- user's label_id is 1;
- knowledge entities's label_id is 2;
### movielens_e.csv
- label_id 0: user - item;
- label_id 1: item - user;
- label_id 2-15: item - knowledge graph entity;
- label_id 16-29: knowledge graph entity - item;


## Yelp
### yelp_v.csv
- item's label_id is 0;
- user's label_id is 1;
- knowledge entities's label_id is 2;
### yelp_e.csv
- label_id 0-41: item - knowledge graph entity;
- label_id 42-83: knowledge graph entity - item;
- label_id 84: user - item;
- label_id 85: item - user;
- label_id 86: user - user;


## CiaoDVD
### ciao_v.csv
- item's label_id is 0;
- user's label_id is 1;
- knowledge entities's label_id is 2;
### ciao_e.csv
- label_id 0: user - item;
- label_id 1: item - user;
- label_id 2: item - knowledge graph entity;
- label_id 3: knowledge graph entity - item;
- label_id 4-6: user - knowledge graph entity;
- label_id 7-9: knowledge graph entity - user;
