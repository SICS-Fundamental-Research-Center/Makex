
import os
import csv

os.environ["DGLBACKEND"] = "pytorch"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import dgl
import copy

class HomoDGLGraph():

    def __init__(self, dataset_name, srcs, dsts, edge_labels):
        self.dataset_name = dataset_name
        self.srcs = srcs
        self.dsts = dsts
        self.edge_labels = edge_labels

    def buildGraph(self):
        g = dgl.DGLGraph()
        n_nodes = 0
        if self.dataset_name == 'dvd':
            n_nodes = 119231
        elif self.dataset_name == 'ciao':
            n_nodes = 106383
        elif self.dataset_name == 'movielens':
            n_nodes = 21201
        elif self.dataset_name == 'yelp':
            n_nodes = 93289
        else:
            print("dataset {} not found".format(self.dataset_name))
        print(n_nodes)
        if self.dataset_name == 'movielens':
            label_1_idx = np.where(self.edge_labels == 1)
            label_3_idx = np.where(self.edge_labels == 3)
            self.edge_labels[label_1_idx] = 0
            self.edge_labels[label_3_idx] = 2

            etypes_dict = {}
            for etype in self.edge_labels:
                if etype in etypes_dict.keys():
                    continue
                else:
                    etypes_dict[etype] = etype

            etypes_dict = dict(sorted(etypes_dict.items()))
            etypes_np = []
            for idx in etypes_dict.keys():
                etypes_np.append(idx)
            
            for i in range(len(etypes_np)):
                real_etype = etypes_np[i]
                self.edge_labels[np.where(self.edge_labels == real_etype)] = i
        

        g.add_nodes(n_nodes)
        
        g.add_edges(self.srcs, self.dsts)
        g.ndata['id'] = torch.arange(n_nodes, dtype=torch.long)
        g.edata['type'] = torch.LongTensor(self.edge_labels)
        g.edata['eid'] = torch.arange(self.edge_labels.shape[0], dtype=torch.long)

        torch.manual_seed(3407)
        g.ndata['feat'] = torch.randn(n_nodes, 16)

        nodes = g.nodes().numpy()
        print("nodes types")
        ntypes = []

        if self.dataset_name == 'yelp':
            items_start_id = 0
            items_end_id = 45537

            users_start_id = 45538
            users_end_id = 91456

            kgs_start_id = 91457
            kgs_end_id = 93289

            for v in nodes:
                if v >= items_start_id and v <= items_end_id:
                    ntypes.append(0)
                if v >= users_start_id and v <= users_end_id:
                    ntypes.append(1)
                if v >= kgs_start_id and v <= kgs_end_id:
                    ntypes.append(2)
        elif self.dataset_name == 'movielens':
            items_start_id = 0
            items_end_id = 3705

            users_start_id = 3706
            users_end_id = 9745

            kgs_start_id = 9746
            kgs_end_id = 21200

            for v in nodes:
                if v >= items_start_id and v <= items_end_id:
                    ntypes.append(0)
                if v >= users_start_id and v <= users_end_id:
                    ntypes.append(1)
                if v >= kgs_start_id and v <= kgs_end_id:
                    ntypes.append(2)
        elif self.dataset_name == 'ciao':
            items_start_id = 0
            items_end_id = 16121

            users_start_id = 16122
            users_end_id = 46565

            kgs_start_id = 46566
            kgs_end_id = 119230

            for v in nodes:
                if v >= items_start_id and v <= items_end_id:
                    ntypes.append(0)
                if v >= users_start_id and v <= users_end_id:
                    ntypes.append(1)
                if v >= kgs_start_id and v <= kgs_end_id:
                    ntypes.append(2)
        else:
            print("node types have not been intialized..")    
        
        g.ndata['ntype'] = torch.from_numpy(np.array(ntypes))
        
        return g

    def process(self):
        G = self.buildGraph()
        return G

class HeteroDGLGraph():

    def __init__(self, dataset_name, origin_graph, edge_labels=None, edge_label_mapping=None):
        self.dataset_name = dataset_name
        self.origin_graph = origin_graph
        self.edge_labels = edge_labels
        self.edge_label_mapping = edge_label_mapping

    def buildGraph(self):

        edges_group  = self.origin_graph.groupby(['etype'])

        hetero_graph_dict = {}
        for label_id in edges_group.groups:
            edges_of_id = edges_group.get_group(label_id)
            srcs = edges_of_id["src"].to_numpy()
            dsts = edges_of_id["dst"].to_numpy()

            edge_type_name = ()
            if self.dataset_name == 'movielens':
                if label_id == 0 or label_id == 1:
                    edge_type_name = ('user','rate', 'item')
                elif label_id ==2 or label_id ==3:
                    edge_type_name = ('item', 'ratedby', 'user')
                elif label_id >3 and label_id < 24:
                    rev_edge_label = self.edge_label_mapping[label_id][1]
                    edge_type_name = ('item', str(self.edge_labels[rev_edge_label][1]), 'kg')

                else:
                    rev_edge_label = self.edge_label_mapping[label_id][1]
                    edge_type_name = ('kg', str(self.edge_labels[rev_edge_label][1]), 'item')
            elif self.dataset_name == 'yelp':
                if label_id == 84:
                    edge_type_name = ('user','rate', 'item')
                elif label_id == 85:
                    edge_type_name = ('item', 'ratedby', 'user')
                elif label_id >=0 and label_id <= 41:
                    edge_type_name = ('item', str(label_id), 'kg')

                elif label_id >= 42 and label_id <= 83:
                    edge_type_name = ('kg', str(label_id), 'item')
                else:
                    edge_type_name = ('user', 'friend', 'user')
            elif self.dataset_name == 'dvd' or self.dataset_name == 'ciao':
                if label_id == 0:
                    edge_type_name = ('user','rate', 'item')
                elif label_id == 1:
                    edge_type_name = ('item', 'ratedby', 'user')
                elif label_id == 2:
                    edge_type_name = ('item', str(label_id), 'kg')
                elif label_id == 3:
                    edge_type_name = ('kg', str(label_id), 'item')
                elif label_id >=4 and label_id <= 6:
                    edge_type_name = ('user', str(label_id), 'kg')

                elif label_id >= 7 and label_id <= 9:
                    edge_type_name = ('kg', str(label_id), 'user')
                else:
                    edge_type_name = ('user', 'friend', 'user')
            if edge_type_name in hetero_graph_dict.keys():
                print("already exist...")
                pre_srcs = hetero_graph_dict[edge_type_name][0].numpy()
                pre_dsts = hetero_graph_dict[edge_type_name][1].numpy()
                srcs = np.concatenate((pre_srcs, srcs), axis=0)
                dsts = np.concatenate((pre_dsts, dsts), axis=0)
                hetero_graph_dict[edge_type_name] = (torch.from_numpy(srcs), torch.from_numpy(dsts))
            else:
                hetero_graph_dict[edge_type_name] = (torch.from_numpy(srcs), torch.from_numpy(dsts))

        G = dgl.heterograph(hetero_graph_dict)
        return G
    
    def add_node_features(self, graph):
        torch.manual_seed(3407)
        new_graph = copy.deepcopy(graph)
        num_items_sg = graph.num_nodes('item') if 'item' in graph.ntypes else 0
        num_users_sg = graph.num_nodes('user') if 'user' in graph.ntypes else 0
        num_kgs_sg = graph.num_nodes('kg') if 'kg' in graph.ntypes else 0
        
        item_feats = torch.randn(num_items_sg, 16)
        user_feats = torch.randn(num_users_sg, 16)

        print(user_feats.shape)
        print(item_feats.shape)
        if 'item' in graph.ntypes:
            new_graph.nodes['item'].data['feat'] = item_feats
        if 'user' in graph.ntypes:
            new_graph.nodes['user'].data['feat'] = user_feats
        if 'kg' in graph.ntypes:
            new_graph.nodes['kg'].data['feat'] = torch.randn(num_kgs_sg, 16)

        return new_graph

    def process(self):

        G = self.buildGraph()

        G = self.add_node_features(G)
        return G

