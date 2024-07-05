import itertools
import os

os.environ["DGLBACKEND"] = "pytorch"

import dgl
import dgl.data

import numpy as np
import scipy.sparse as sp
from model import GraphSAGE,DotPredictor,MLPPredictor

import torch
import csv
import random as rd
import pandas as pd


class dataLoader():
    def __init__(self, DGLgraph, test_ratio=0.1, seed=1024):
        self.g = DGLgraph
        self.test_ratio = test_ratio
        rd.seed(seed)

        u, v = self.g.edges()

        eids = np.arange(self.g.num_edges())
        self.eids = np.random.permutation(eids)
        self.test_size = int(len(self.eids) * self.test_ratio)
        self.train_size = self.g.num_edges() - self.test_size
        self.test_pos_u, self.test_pos_v = u[self.eids[:self.test_size]], v[self.eids[:self.test_size]]
        self.train_pos_u, self.train_pos_v = u[self.eids[self.test_size:]], v[self.eids[self.test_size:]]

        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
        adj_neg = 1 - adj.todense() - np.eye(self.g.num_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_eids = np.random.choice(len(neg_u), self.g.num_edges())
        self.test_neg_u, self.test_neg_v = (
            neg_u[neg_eids[:self.test_size]],
            neg_v[neg_eids[:self.test_size]],
        )
        self.train_neg_u, self.train_neg_v = (
            neg_u[neg_eids[self.test_size:]],
            neg_v[neg_eids[self.test_size:]],
        )

    def return_train_g(self):
        self.train_g = dgl.remove_edges(self.g, self.eids[:self.test_size])

        self.train_pos_g = dgl.graph((self.train_pos_u, self.train_pos_v), num_nodes=self.g.num_nodes())
        self.train_neg_g = dgl.graph((self.train_neg_u, self.train_neg_v), num_nodes=self.g.num_nodes())

        self.test_pos_g = dgl.graph((self.test_pos_u, self.test_pos_v), num_nodes=self.g.num_nodes())
        self.test_neg_g = dgl.graph((self.test_neg_u, self.test_neg_v), num_nodes=self.g.num_nodes())

        return self.train_g, self.train_pos_g, self.train_neg_g, self.test_pos_g, self.test_neg_g
    


class heteroDataLoader():
    def __init__(self, DGLgraph, test_ratio=0.2):
        self.g = DGLgraph
        self.test_ratio = test_ratio

    
    def load_data(self, train_filename, test_filename):
        train_filename_pos = train_filename + '_pos.csv'
        train_filename_neg = train_filename + '_neg.csv'
        test_filename_pos = test_filename + '_pos.csv'
        test_filename_neg = test_filename + '_neg.csv'

        if os.path.isfile(train_filename_pos):
            # load the data from files
            print("loading data from local")
            np_train_pos = pd.read_csv(train_filename_pos).to_numpy()
            np_train_neg = pd.read_csv(train_filename_neg).to_numpy()
            np_test_pos = pd.read_csv(test_filename_pos).to_numpy()
            np_test_neg = pd.read_csv(test_filename_neg).to_numpy()

            self.train_pos_u, self.train_pos_v = np_train_pos[:,0], np_train_pos[:,1]
            self.test_pos_u, self.test_pos_v  = np_test_pos[:,0], np_test_pos[:,1]

            self.train_neg_u, self.train_neg_v  = np_train_neg[:,0], np_train_neg[:,1]
            self.test_neg_u, self.test_neg_v  = np_test_neg[:,0], np_test_neg[:,1]

        else:
            print("spliting data to train and test")
            u, v = self.g.edges(etype=('user', 'rate', 'item'))
            u = u.numpy()
            v = v.numpy()
            self.num_users = self.g.number_of_nodes('user')
            self.num_items = self.g.number_of_nodes('item')
            print(self.num_users, self.num_items)

            ui_num_edges = self.g.num_edges(etype=('user', 'rate', 'item'))
            ui_eids = np.arange(ui_num_edges)
            self.ui_eids = np.random.permutation(ui_eids)
            self.test_size = int(len(self.ui_eids) * self.test_ratio)
            self.train_size = ui_num_edges  - self.test_size
            self.test_pos_u, self.test_pos_v = u[self.ui_eids[:self.test_size]], v[self.ui_eids[:self.test_size]]
            self.train_pos_u, self.train_pos_v = u[self.ui_eids[self.test_size:]], v[self.ui_eids[self.test_size:]]

            train_neg_u = []
            train_neg_v = []
            print("finding negative sample for train set")
            for u in self.train_pos_u:
                neighs_u = self.g.successors(u, etype=('user', 'rate', 'item')).numpy()
                while True:
                    neg_i_id = rd.choice(range(self.num_items))
                    if neg_i_id not in neighs_u: break
                train_neg_u.append(u)
                train_neg_v.append(neg_i_id)
            
            self.train_neg_u = train_neg_u
            self.train_neg_v = train_neg_v

            test_neg_u = []
            test_neg_v = []

            print("finding negative sample for test set")
            for u in self.test_pos_u:
                neighs_u = self.g.successors(u, etype=('user', 'rate', 'item')).numpy()
                while True:
                    neg_i_id = rd.choice(range(self.num_items))
                    if neg_i_id not in neighs_u: break
                test_neg_u.append(u)
                test_neg_v.append(neg_i_id)
            
            self.test_neg_u = test_neg_u
            self.test_neg_v = test_neg_v

            print("saving files")
            

            with open(train_filename_pos, "w") as train_pos:
                temp_writer = csv.DictWriter(train_pos, fieldnames=['src', 'dst', 'label'])
                for idx in range(len(self.train_pos_u)):
                    tmp_dic = {
                        "src": self.train_pos_u[idx],
                        "dst": self.train_pos_v[idx],
                        "label": 1
                    }
                    temp_writer.writerow(tmp_dic)
            
            print("train pos done.")
            with open(train_filename_neg, "w") as train_neg:
                temp_writer = csv.DictWriter(train_neg, fieldnames=['src', 'dst', 'label'])
                for idx in range(len(self.train_neg_u)):
                    tmp_dic = {
                        "src": self.train_neg_u[idx],
                        "dst": self.train_neg_v[idx],
                        "label": 0
                    }
                    temp_writer.writerow(tmp_dic)
            
            print("train neg done.")

            with open(test_filename_pos, "w") as test_pos:
                temp_writer = csv.DictWriter(test_pos, fieldnames=['src', 'dst', 'label'])
                for idx in range(len(self.test_pos_u)):
                    tmp_dic = {
                        "src": self.test_pos_u[idx],
                        "dst": self.test_pos_v[idx],
                        "label": 1
                    }
                    temp_writer.writerow(tmp_dic)
            print("test pos done.")

            with open(test_filename_neg, "w") as test_neg:
                temp_writer = csv.DictWriter(test_neg, fieldnames=['src', 'dst', 'label'])
                for idx in range(len(self.test_neg_u)):
                    tmp_dic = {
                        "src": self.test_neg_u[idx],
                        "dst": self.test_neg_v[idx],
                        "label": 0
                    }
                    temp_writer.writerow(tmp_dic)

            print("test neg done ....")
    
    def return_train_g(self):
        test_eids = self.g.edge_ids(self.test_pos_u, self.test_pos_v, etype=('user', 'rate', 'item'))
        test_rev_eids = self.g.edge_ids(self.test_pos_v, self.test_pos_u, etype=('item', 'ratedby', 'user'))
        self.train_g = dgl.remove_edges(self.g, test_eids, etype=('user', 'rate', 'item'))
        self.train_g = dgl.remove_edges(self.train_g, test_rev_eids, etype=('item', 'ratedby', 'user'))
        num_nodes_dict = {'user': self.g.num_nodes('user'), 'item': self.g.num_nodes('item')}
        self.train_pos_g = dgl.heterograph({
                ('user', 'rate', 'item'): (self.train_pos_u, self.train_pos_v)
            }, num_nodes_dict=num_nodes_dict)
        self.train_neg_g = dgl.heterograph({
                ('user', 'rate', 'item'): (self.train_neg_u, self.train_neg_v)
            }, num_nodes_dict=num_nodes_dict)

        self.test_pos_g = dgl.heterograph({
                ('user', 'rate', 'item'): (self.test_pos_u, self.test_pos_v)
            }, num_nodes_dict=num_nodes_dict)
        
        self.test_neg_g = dgl.heterograph({
                ('user', 'rate', 'item'): (self.test_neg_u, self.test_neg_v)
            }, num_nodes_dict=num_nodes_dict)
        
        pos_ui_num_edges = self.train_pos_g.num_edges(etype=('user', 'rate', 'item'))
        pos_ui_eids = np.arange(pos_ui_num_edges)
        sel_pos_ui_eids = np.random.permutation(pos_ui_eids)
        self.val_pos_size = int(len(sel_pos_ui_eids) * 0.1)
        self.val_pos_u, self.val_pos_v = self.train_pos_u[sel_pos_ui_eids[:self.val_pos_size]], self.train_pos_v[sel_pos_ui_eids[:self.val_pos_size]]

        self.val_pos_g = dgl.heterograph({
                ('user', 'rate', 'item'): (self.val_pos_u, self.val_pos_v)
            }, num_nodes_dict=num_nodes_dict)

        neg_ui_num_edges = self.train_neg_g.num_edges(etype=('user', 'rate', 'item'))
        neg_ui_eids = np.arange(neg_ui_num_edges)
        sel_neg_ui_eids = np.random.permutation(neg_ui_eids)
        self.val_neg_size = int(len(sel_neg_ui_eids) * 0.1)
        self.val_neg_u, self.val_neg_v = self.train_neg_u[sel_neg_ui_eids[:self.val_neg_size]], self.train_neg_v[sel_neg_ui_eids[:self.val_neg_size]]

        self.val_neg_g = dgl.heterograph({
                ('user', 'rate', 'item'): (self.val_neg_u, self.val_neg_v)
            }, num_nodes_dict=num_nodes_dict)


        return self.train_g, self.train_pos_g, self.train_neg_g, self.test_pos_g, self.test_neg_g, self.val_pos_g, self.val_neg_g
    

    def return_train_g_for_kgat(self):
        test_eids = self.g.edge_ids(self.test_pos_u, self.test_pos_v)
        test_rev_eids = self.g.edge_ids(self.test_pos_v, self.test_pos_u)
        self.train_g = dgl.remove_edges(self.g, test_eids)
        self.train_g = dgl.remove_edges(self.train_g, test_rev_eids)

        self.train_pos_g = dgl.graph((self.train_pos_u, self.train_pos_v))
        self.train_neg_g = dgl.graph((self.train_neg_u, self.train_neg_v))

        self.test_pos_g = dgl.graph((self.test_pos_u, self.test_pos_v))
        
        self.test_neg_g = dgl.graph((self.test_neg_u, self.test_neg_v))
        
        pos_ui_num_edges = self.train_pos_g.num_edges()
        pos_ui_eids = np.arange(pos_ui_num_edges)
        sel_pos_ui_eids = np.random.permutation(pos_ui_eids)
        self.val_pos_size = int(len(sel_pos_ui_eids) * 0.1)
        self.val_pos_u, self.val_pos_v = self.train_pos_u[sel_pos_ui_eids[:self.val_pos_size]], self.train_pos_v[sel_pos_ui_eids[:self.val_pos_size]]

        self.val_pos_g = dgl.graph((self.val_pos_u, self.val_pos_v))

        neg_ui_num_edges = self.train_neg_g.num_edges()
        neg_ui_eids = np.arange(neg_ui_num_edges)
        sel_neg_ui_eids = np.random.permutation(neg_ui_eids)
        self.val_neg_size = int(len(sel_neg_ui_eids) * 0.1)
        self.val_neg_u, self.val_neg_v = self.train_neg_u[sel_neg_ui_eids[:self.val_neg_size]], self.train_neg_v[sel_neg_ui_eids[:self.val_neg_size]]

        self.val_neg_g = dgl.graph((self.val_neg_u, self.val_neg_v))

        return self.train_g, self.train_pos_g, self.train_neg_g, self.test_pos_g, self.test_neg_g, self.val_pos_g, self.val_neg_g
    
