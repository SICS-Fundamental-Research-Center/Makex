# for each explanation (subgraph), as input
# output the prediction of a pair node


import os
import sys 
# sys.path.append("..")

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

import time


# from ..data_split import heteroDataLoader
from model import ScorePredictor,DotPredictor
# from metrics import metricsTrain
from data_split import heteroDataLoader
from build_heteroGraph import HeteroDGLGraph,HomoDGLGraph
from adapt_sx import KGATSX

import csv

sys.path.append("metrics")

def find_subgraph(dgl_graph, pivots, num_hops=1):
    # find a subgraph based on the pair 
    # return both subgraph and the number of nodes in this sg

    sg, inverse_indices = dgl.khop_out_subgraph(dgl_graph, pivots, k=num_hops, relabel_nodes=True, store_ids=False)
    num_nodes = sg.number_of_nodes()
    return sg, num_nodes, inverse_indices

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("loading dataset")
file_location = "../../dataset/"
dataset_name = "yelp"
dir_to_load = file_location + dataset_name
graph_filename = "/" + dataset_name +"_e.csv"
edge_label_filename = "/attributes_info/"+ dataset_name + "_kg_label_map.txt"
node_feature_filename = "/feature_vectors_v.csv"
# train_filename =  dir_to_load + "/train"
# test_filename = dir_to_load + "/test"

homoGraph = HomoDGLGraph(dir_to_load, dataset_name, graph_filename, node_feature_filename)
G = homoGraph.process()
print("creating train and test data")

G = G.to(device)


# load the trained model
print("loading the model")
# file_location = "../../dataset/"
modelname = 'kgat'
model_name = "/models/kgat.pt"
model_dir_to_load = dir_to_load + model_name
load_model = torch.load(model_dir_to_load)
print("model loaded")

thre_filename = '/models/'+  modelname + '_threshold.csv'

threshold = np.genfromtxt(dir_to_load + "/" + thre_filename, delimiter=',', dtype=None)[0]


pairs_filename = "/models/" + modelname + "_sx_pairs.csv"
pairs_csv_dtype = {'user_id': np.int64, 'item_id': np.int64}
pairs_fieldnames=['user_id', 'item_id']
eval_pairs = pd.read_csv(dir_to_load + pairs_filename, names=pairs_fieldnames, dtype=pairs_csv_dtype, header=1).to_numpy()
print(eval_pairs.shape)


explain_eval_dict ={}
i=0

dir_to_save = file_location + "/" + dataset_name + "/" + 'sx_result/for_patterns/' + modelname + '/'
gnn_exp_sg_filename = dir_to_save + '/subgraphs_eids.csv'
gnn_exp_pairs_filename = dir_to_save + '/subgraphs_pair.csv'
gnn_exp_time_filename = dir_to_save + "/subgraphs_time_largeuser.csv"

total_time = 0


i = 0
min_num_nodes_exp = 150
all_feat_vec = []
all_eids_vec = []
all_pairs_vec = []
for pair in eval_pairs:
    user = pair[0]
    item = pair[1]

    pivots = [user, item]
    all_pairs_vec.append(pivots)
    print(i,pivots)

    SG, num_nodes_in_sg, inverse_indices = find_subgraph(G, pivots, 2)
    node_min = min(int(num_nodes_in_sg / 2), min_num_nodes_exp) 
    print(inverse_indices)
    new_pivots = inverse_indices.cpu().detach().numpy()
    start_time = time.time()
    explainer = KGATSX(load_model, num_hops=2, num_rollouts=5, num_child=5, node_min=int(node_min), shapley_steps=20)
    nodes_explain = explainer.explain_graph(SG, new_pivots, SG.ndata['feat'], target_class=1)
    end_time = time.time()
    print("# nodes for explaination {}".format(len(nodes_explain)))
    print(nodes_explain)
    nodes_explain = nodes_explain.cpu().detach().numpy()

    all_eids_vec.append(nodes_explain)

    end_time = time.time()
    total_time = total_time + (end_time - start_time)

    i = i +1 
    if i > 50:
        break
 

sg_file_df = pd.DataFrame(all_eids_vec)
sg_file_df.to_csv(gnn_exp_sg_filename)

pairs_file_df = pd.DataFrame(all_pairs_vec)
pairs_file_df.to_csv(gnn_exp_pairs_filename)

# print(sorted_dict)
with open(gnn_exp_time_filename, "w") as exp_time_file:
    time_writer = csv.DictWriter(exp_time_file, fieldnames=['total_time', 'avg_time'])

    tmp_time_dic = {
                    "total_time": total_time,
                    'avg_time': total_time / float(i)
                }
    time_writer.writerow(tmp_time_dic)