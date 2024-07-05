
import os
import sys 
sys.path.append("..")

os.environ["DGLBACKEND"] = "pytorch"

import pandas as pd
import dgl
import dgl.data
import dgl.dataloading

import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import scipy.sparse as sp

from build_graph_for_eval import HomoDGLGraph,HeteroDGLGraph
from model import ScorePredictor,DotPredictor

import csv

def gnn_infer(dir_to_load, dataset_name, modelname, train_g, pivot_x, pivot_y, threshold):
    model_filename = '/models/'+ modelname + '.pt'
    model = torch.load(dir_to_load + model_filename)
    model.eval()
    print("model loaded")
    
    pred = ScorePredictor() if modelname == 'pinsage' else DotPredictor()
    h={}
    if modelname == 'hgt':
        h = model(train_g, train_g.ndata['feat'], train_g.ndata['ntype'], train_g.edata['type'])
    elif modelname == 'pinsage':
        if 'user' not in train_g.ntypes or 'item' not in train_g.ntypes:
            return 0,0
        input_features = train_g.srcdata['feat']
        if (type(input_features) is dict) == False:
            input_features={}
            if 'user' in train_g.ntypes:
                input_features['user'] = train_g.nodes['user'].data['feat']
            if 'item' in train_g.ntypes:
                input_features['item'] = train_g.nodes['item'].data['feat']
            if 'kg' in train_g.ntypes:
                input_features['kg'] = train_g.nodes['kg'].data['feat']
        elif 'user' not in input_features:
            input_features['user'] = train_g.nodes['user'].data['feat']
        elif 'item' not in input_features:
            input_features['item'] = train_g.nodes['item'].data['feat']
        elif 'kg' not in input_features and 'kg' in train_g.ntypes:
            input_features['kg'] = train_g.nodes['kg'].data['feat']

        print(input_features)
        blocks = [train_g, train_g]
        h = model(blocks, input_features)
        print(h)
        if 'user' in h.keys() and 'item' in h.keys():
            h = {
                "user": h['user'],
                "item": h['item']
            }
        else:
            return 0,0
    elif modelname == 'kgat':
        with torch.no_grad():
            print("Compute attention weight in eval func ...")
            print(train_g.edges())
            A_w = model.compute_attention(train_g)
            train_g.edata['w'] = A_w
            h = model.gnn(train_g, train_g.ndata['feat'])
            n_nodes = 9746 if dataset_name == 'movielens' else 91457
            h = h[:n_nodes,:]
    else:
        print("no such model")
    pivot_score = None
    if modelname == 'kgat' or modelname == 'hgt':
        print(h)
        print(pivot_x, pivot_y)
        print("constructing a new graph")
        pivot_g = dgl.graph(([pivot_x], [pivot_y]))
        max_id = max(pivot_x, pivot_y)
        print(max_id)
        pivot_score = pred(pivot_g.to(device), h[:max_id+1, :]).cpu().detach().numpy()
    else:

        num_nodes_dict = {'user': train_g.num_nodes('user'), 'item': train_g.num_nodes('item')}
        pivot_g = dgl.heterograph({
                ('user', 'rate', 'item'): ([pivot_x], [pivot_y])
            }, num_nodes_dict=num_nodes_dict)
        print(h)
        print(pivot_g)
        pivot_score = pred(pivot_g.to(device), h).cpu().detach().numpy()
        print(pivot_score)

    rec = 0
    if pivot_score[0][0] >= threshold:
        rec = 1

    return pivot_score[0][0], rec


print("loading dataset")

print("the dataset...")
file_location = "../../../dataset"
print('Enter dataset name:')
dataset_name = input()
print('Enter the exact folder name:')
res_dir = input()

print('Enter gnn model name:')
modelname = input()


res_filename =  'makex_result/' + modelname + '/' + res_dir + '/'
dir_to_load = file_location + "/" + dataset_name + "/" 

up_dir_to_load = file_location + "/" + dataset_name + "/" + res_filename

sub_dirs = os.listdir(up_dir_to_load)

thre_filename = '/models/'+  modelname + '_threshold.csv'

threshold = np.genfromtxt(file_location + "/" + dataset_name + "/" + thre_filename, delimiter=',', dtype=None)[0]

if torch.cuda.is_available():
    if modelname == 'kgat':
        device = torch.device('cuda:2')
    elif modelname == 'hgt':
        device = torch.device('cuda:1')
    elif modelname == 'pinsage':
        device = torch.device('cuda:1')
else:
    device = 'cpu'

for sub_dir in sub_dirs:
    resdir_to_load = up_dir_to_load + sub_dir + '/'
    print(resdir_to_load)
    if os.path.isdir(resdir_to_load) == False:
        continue


    graph_filename = 'e.csv'
    node_filename ='/v.csv'
    attr_filename = "/feature_vectors_v.csv"

    makex_feat = pd.read_csv(resdir_to_load + attr_filename).to_numpy()

    makex_sgs = pd.read_csv(resdir_to_load + graph_filename)
    columns_to_convert = ['pair_id', 'pivot_x', 'pivot_y', 'topk']
    makex_sgs[columns_to_convert] = makex_sgs[columns_to_convert].astype(int)

    makex_nodes = pd.read_csv(resdir_to_load + node_filename) 
    columns_to_convert_2 = ['pair_id', 'pivot_x', 'pivot_y', 'topk', 'vertex_id', 'label_id']
    makex_nodes[columns_to_convert] = makex_nodes[columns_to_convert].astype(int)


    sgs_group = makex_sgs.groupby(['pair_id','topk'])
    nodes_group = makex_nodes.groupby(['pair_id','topk'])


    feat_start_idx = 0

    explain_eval_dict = {}
    i=0
    for pair_id, topk in sgs_group.groups:
        edges_of_id = sgs_group.get_group((pair_id, topk))
        nodes_of_id = nodes_group.get_group((pair_id, topk))
        
        SG = None

        if modelname == 'kgat' or modelname == 'hgt':
            pivot_x = edges_of_id["pivot_x"].to_numpy()
            pivot_y = edges_of_id['pivot_y'].to_numpy()
            src = edges_of_id["source_id:int"].to_numpy()
            dst = edges_of_id["target_id:int"].to_numpy()
            edge_labels = edges_of_id['label_id:int'].to_numpy()

            nodes = nodes_of_id['vertex_id'].to_numpy() 
            feat_end_idx = feat_start_idx + nodes.size
            node_feature = makex_feat[feat_start_idx:feat_end_idx, :]
            feat_start_idx = feat_end_idx

            homoGraph = HomoDGLGraph(dataset_name, src, dst, edge_labels, node_feature, nodes)
            SG = homoGraph.process()

            SG = SG.to(device)
            print(SG)
        else:

            pivot_x = edges_of_id["pivot_x"].to_numpy()
            pivot_y = edges_of_id['pivot_y'].to_numpy()

            nodes = nodes_of_id['vertex_id'].to_numpy() 
            print(nodes)
            feat_end_idx = feat_start_idx + nodes.size
            node_feature = makex_feat[feat_start_idx:feat_end_idx, :]
            feat_start_idx = feat_end_idx
            heteroGraph = HeteroDGLGraph(dataset_name, edges_of_id, node_feature, nodes)
            SG = heteroGraph.process()

            SG = SG.to(device)

            print(SG)
        
        pred_score, rec = gnn_infer(dir_to_load, dataset_name, modelname, SG, pivot_x[0], pivot_y[0],threshold)

        if (pair_id, topk) in explain_eval_dict.keys():
            print("%s exists " %(pair_id, topk))
        else:
            explain_eval_dict[(pair_id, topk)] = (pred_score, rec)

        i = i + 1

    print("this folder has {} pairs".format(i))
    sorted_dict = dict(sorted(explain_eval_dict.items()))

    eval_csv_columns = ['pair_id','topk','pred_score', 'rec']
    eval_filename = "eval.csv"
    with open(resdir_to_load + eval_filename, 'w') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=eval_csv_columns)
            for key in sorted_dict.keys():
                temp_dict = {
                    'pair_id': key[0],
                    'topk': key[1],
                    'pred_score': sorted_dict[key][0], 
                    'rec': sorted_dict[key][1]
                }
                writer.writerow(temp_dict)
