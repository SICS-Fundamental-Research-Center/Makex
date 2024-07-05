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

    def __init__(self, dir_to_load, dataset_name, graph_filename, node_feature_filename):
        self.dir_to_load = dir_to_load
        self.dataset_name = dataset_name
        self.graph_filename = graph_filename
        
        self.node_feature_filename = node_feature_filename

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
        srcs = self.origin_graph["source_id:int"].to_numpy()
        dsts = self.origin_graph["target_id:int"].to_numpy()
        edge_labels = self.origin_graph['label_id:int'].to_numpy()
        if self.dataset_name == 'movielens':
            label_1_idx = np.where(edge_labels == 1)
            label_3_idx = np.where(edge_labels == 3)
            edge_labels[label_1_idx] = 0
            edge_labels[label_3_idx] = 2

            etypes_dict = {}
            for etype in edge_labels:
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
                edge_labels[np.where(edge_labels == real_etype)] = i
        

        g.add_nodes(n_nodes)
        
        g.add_edges(srcs, dsts)
        g.ndata['id'] = torch.arange(n_nodes, dtype=torch.long)
        g.edata['type'] = torch.LongTensor(edge_labels)
        g.edata['eid'] = torch.arange(edge_labels.shape[0], dtype=torch.long)

        nodes = g.nodes().numpy()
        print("nodes types")
        ntypes = []
        if self.dataset_name == 'dvd':

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
            
        elif self.dataset_name == 'ciao':
            items_start_id = 0
            items_end_id = 16119

            users_start_id = 16120
            users_end_id = 33717

            kgs_start_id = 33718
            kgs_end_id = 106382

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
        elif self.dataset_name == 'yelp':
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
                    
        g.ndata['ntype'] = torch.from_numpy(np.array(ntypes))
        return g
    
    def add_node_features(self, graph):
        torch.manual_seed(3407)
        new_graph = copy.deepcopy(graph)

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

        if self.dataset_name == 'dvd':
            n_items = self.node_features.shape[0]
            n_users_plus_kg = n_nodes - n_items

            user_kg_feats = torch.randn(n_users_plus_kg, 16)
            item_feats = torch.from_numpy(self.node_features[:,1:]).to(torch.float32)
            new_graph.ndata['feat'] = torch.cat((item_feats, user_kg_feats), 0)
        else:
            n_users_plus_items = self.node_features.shape[0]
            n_kgs = n_nodes - n_users_plus_items
            kg_feats = torch.randn(n_kgs, 16)
            cf_feats = torch.from_numpy(self.node_features[:,1:]).to(torch.float32)
            print(cf_feats.shape)

            new_graph.ndata['feat'] = torch.cat((cf_feats, kg_feats), 0)

        return new_graph

    def process(self):
        self.origin_graph = pd.read_csv(self.dir_to_load + self.graph_filename)
        self.node_features = pd.read_csv(self.dir_to_load + self.node_feature_filename).to_numpy()

        G = self.buildGraph()
        G = self.add_node_features(G)
        return G

class HeteroDGLGraph():

    def __init__(self, dir_to_load, dataset_name, graph_filename, edge_label_filename, node_feature_filename):
        self.dir_to_load = dir_to_load
        self.graph_filename = graph_filename
        self.dataset_name = dataset_name
        
        self.edge_label_filename = edge_label_filename
        self.node_feature_filename = node_feature_filename

    
    def buildGraph(self):
        

        edges_group  = self.origin_graph.groupby(['label_id:int'])

        hetero_graph_dict = {}
        for label_id in edges_group.groups:
            edges_of_id = edges_group.get_group(label_id)
            srcs = edges_of_id["source_id:int"].to_numpy()
            dsts = edges_of_id["target_id:int"].to_numpy()

            edge_type_name = ()
            if self.dataset_name == 'movielens':
                if label_id == 0 or label_id == 1:
                    edge_type_name = ('user','rate', 'item')
                elif label_id ==2 or label_id ==3:
                    edge_type_name = ('item', 'ratedby', 'user')
                elif label_id >3 and label_id < 24:
                    edge_type_name = ('item', str(self.edge_labels[label_id][1]), 'kg')

                else:
                    edge_type_name = ('kg', str(self.edge_labels[label_id][1]), 'item')
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

        if self.dataset_name == 'dvd':
            num_items = 16122
            num_users = 46565
            num_kgs = 119231

            new_graph.nodes['item'].data['feat'] = torch.from_numpy(self.node_features[:,1:]).to(torch.float32)
            new_graph.nodes['user'].data['feat'] = torch.randn(num_users, 16)
            new_graph.nodes['kg'].data['feat'] = torch.randn(num_kgs, 16)
        if self.dataset_name == 'ciao':
            num_items = 16120
            num_users = 33718
            num_kgs = 106383

            new_graph.nodes['item'].data['feat'] = torch.from_numpy(self.node_features[:,1:]).to(torch.float32)
            new_graph.nodes['user'].data['feat'] = torch.randn(num_users, 16)
            new_graph.nodes['kg'].data['feat'] = torch.randn(num_kgs, 16)
        else:
            num_items = 3706 if self.dataset_name == 'movielens' else 45538
            num_kgs = 21201 if self.dataset_name == 'movielens' else 93289
            new_graph.nodes['item'].data['feat'] = torch.from_numpy(self.node_features[:num_items,1:]).to(torch.float32)
            new_graph.nodes['user'].data['feat'] = torch.from_numpy(self.node_features[:,1:]).to(torch.float32)
            new_graph.nodes['kg'].data['feat'] = torch.randn(num_kgs, 16)

        return new_graph

    def process(self):
        self.origin_graph = pd.read_csv(self.dir_to_load + self.graph_filename)
        self.edge_labels = pd.read_csv(self.dir_to_load + self.edge_label_filename).to_numpy()
        self.node_features = pd.read_csv(self.dir_to_load + self.node_feature_filename).to_numpy()

        G = self.buildGraph()

        G = self.add_node_features(G)
        
        return G

