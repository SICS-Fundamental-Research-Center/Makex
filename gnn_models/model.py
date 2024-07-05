import itertools
import os

os.environ["DGLBACKEND"] = "pytorch"

import dgl
import dgl.data
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv,RelGraphConv,HGTConv
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.softmax import edge_softmax

import math


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class HGT2(nn.Module):
    def __init__(self, in_size, head_size, num_heads, num_ntypes, num_etypes, dropout=0.2, use_norm=False):
        super(HGT2, self).__init__()
        self.conv1 = HGTConv(in_size, head_size, num_heads, num_ntypes, num_etypes, dropout, use_norm)
        
    def forward(self, g, in_feat,  ntype, etype):

        h = self.conv1(g, in_feat, ntype, etype)
        return h
    
class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(in_feat, hidden_feat, norm='right')
                for rel in rel_names
            })
        self.conv2 = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(hidden_feat, out_feat, norm='right')
                for rel in rel_names
            })

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        return x
    
class RGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,
                 etypes):
        super().__init__()
        self.rgcn = StochasticTwoLayerRGCN(
            in_features, hidden_features, out_features, etypes)
        self.pred = ScorePredictor()

    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.rgcn(blocks, x)
        pos_score = self.pred(positive_graph, x)
        neg_score = self.pred(negative_graph, x)
        return pos_score, neg_score

class PinSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, graph):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        conv_dict_1 = {
            rel[1]: SAGEConv(in_features, hidden_features, aggregator_type='mean') for rel in graph.canonical_etypes
        }

        conv_dict_2 = {
            rel[1]: SAGEConv(hidden_features, out_features, aggregator_type='mean') for rel in graph.canonical_etypes
        }
        self.RSAGEConv_1 = dglnn.HeteroGraphConv(conv_dict_1, aggregate='mean')
        self.RSAGEConv_2 = dglnn.HeteroGraphConv(conv_dict_2, aggregate='mean')

    def forward(self, blocks, x, etype=None):
        h = self.RSAGEConv_1(blocks[0], x)
        h = self.RSAGEConv_2(blocks[1], h)
        return h

class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
            
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        etype = edges.data['id'][0]
        relation_att = self.relation_att[etype]
        relation_pri = self.relation_pri[etype]
        relation_msg = self.relation_msg[etype]
        key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
        att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
        val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
        return {'a': att, 'v': val}
    
    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}
        
    def forward(self, G, inp_key, out_key):
        node_dict, edge_dict = G.node_dict, G.edge_dict
        for srctype, etype, dsttype in G.canonical_etypes:
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]] 
            q_linear = self.q_linears[node_dict[dsttype]]
            

            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
            G.apply_edges(func=self.edge_attention, etype=(srctype, etype, dsttype))
        G.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer = 'mean')
        for ntype in G.ntypes:
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'])
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[n_id](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)
    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
                
class HGT(nn.Module):
    def __init__(self, G, n_inp, n_hid, n_out, n_layers, n_heads, device, use_norm = True):
        super(HGT, self).__init__()
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.device = device
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(G.node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp,   n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, len(G.node_dict), len(G.edge_dict), n_heads, use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G):
        for ntype in G.ntypes:
            n_id = G.node_dict[ntype]
            G.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['feat'].to(self.device)))
        for i in range(self.n_layers):
            self.gcs[i](G, 'h', 'h')
        
        e_user =self.out(G.nodes['user'].data['h'])
        e_item =self.out(G.nodes['item'].data['h'])
        final_h = {
            "user": e_user,
            "item": e_item
        }
        return final_h
    
    def __repr__(self):
        return '{}(n_inp={}, n_hid={}, n_out={}, n_layers={})'.format(
            self.__class__.__name__, self.n_inp, self.n_hid,
            self.n_out, self.n_layers)

# kgat model
def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

def bmm_maybe_select(A, B, index):
    if A.dtype == torch.int64 and len(A.shape) == 1:
        B = B.view(-1, B.shape[2])
        flatidx = index * B.shape[1] + A
        return B.index_select(0, flatidx)
    else:
        BB = B.index_select(0, index)
        return torch.bmm(A.unsqueeze(1), BB).squeeze()

class KGATConv(nn.Module):
    def __init__(self, entity_in_feats, out_feats, dropout, res_type="Bi"):
        super(KGATConv, self).__init__()
        self.mess_drop = nn.Dropout(dropout)
        self._res_type = res_type
        if res_type == "Bi":
            self.res_fc_2 = nn.Linear(entity_in_feats, out_feats, bias=False)
        else:
            raise NotImplementedError

    def forward(self, g, nfeat):
        g = g.local_var()
        g.ndata['h'] = nfeat
        g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_neighbor'))
        h_neighbor = g.ndata['h_neighbor']
        if self._res_type == "Bi":
            out = F.leaky_relu(self.res_fc_2(torch.mul(g.ndata['h'], h_neighbor)))
        else:
            raise NotImplementedError

        return self.mess_drop(out)

class KGAT(nn.Module):
    def __init__(self, node_features, input_node_dim, num_gnn_layers, n_hidden, dropout, use_attention=True,
                 n_entities=None, n_relations=None, relation_dim=None,
                 reg_lambda_kg=0.00001, reg_lambda_gnn=0.00001, res_type="Bi"):
        super(KGAT, self).__init__()
        self._use_KG = True
        self._n_entities = n_entities
        self._n_relations = n_relations
        self._use_attention = use_attention
        self._reg_lambda_kg = reg_lambda_kg
        self._reg_lambda_gnn = reg_lambda_gnn

        self.entity_embed =  nn.Embedding(n_entities, input_node_dim) #
        self.relation_embed = nn.Embedding(n_relations, relation_dim)  #
        self.W_R = nn.Parameter(torch.Tensor(n_relations, input_node_dim, relation_dim))  
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        self.layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            r = int(math.pow(2, i))
            act = None if i+1 == num_gnn_layers else F.relu
            if i==0:
                self.layers.append(KGATConv(input_node_dim, n_hidden // r, dropout))
                
            else:
                r2 = int(math.pow(2, i - 1))
                self.layers.append(KGATConv(n_hidden // r2, n_hidden // r, dropout))


    def transR(self, h, r, pos_t, neg_t):
        h_embed = self.entity_embed(h)  
        r_embed = self.relation_embed(r)
        pos_t_embed = self.entity_embed(pos_t)
        neg_t_embed = self.entity_embed(neg_t)

        h_vec = F.normalize(bmm_maybe_select(h_embed, self.W_R, r), p=2, dim=1)
        r_vec = F.normalize(r_embed, p=2, dim=1)
        pos_t_vec = F.normalize(bmm_maybe_select(pos_t_embed, self.W_R, r), p=2, dim=1)
        neg_t_vec = F.normalize(bmm_maybe_select(neg_t_embed, self.W_R, r), p=2, dim=1)

        pos_score = torch.sum(torch.pow(h_vec + r_vec - pos_t_vec, 2), dim=1, keepdim=True)
        neg_score = torch.sum(torch.pow(h_vec + r_vec - neg_t_vec, 2), dim=1, keepdim=True)
        l = (-1.0) * F.logsigmoid(neg_score-pos_score)
        l = torch.mean(l)
        reg_loss = _L2_loss_mean(h_vec) + _L2_loss_mean(r_vec) + \
                   _L2_loss_mean(pos_t_vec) + _L2_loss_mean(neg_t_vec)
        loss = l + self._reg_lambda_kg * reg_loss
        return loss

    def _att_score(self, edges):
        src_embed = self.entity_embed(edges.src['id'])
        dst_embed = self.entity_embed(edges.dst['id'])
        t_r = torch.matmul(src_embed, self.W_r) 
        h_r = torch.matmul(dst_embed, self.W_r) 
        att_w = torch.bmm(t_r.unsqueeze(1),
                       torch.tanh(h_r + self.relation_embed(edges.data['type'])).unsqueeze(2)).squeeze(-1)
        
        return {'att_w': att_w}
    

    def compute_attention(self, g):
        g = g.local_var()
        for i in range(self._n_relations):
            e_idxs = g.filter_edges(lambda edges: edges.data['type'] ==i)
            self.W_r = self.W_R[i]
            g.apply_edges(self._att_score, e_idxs)
        
        w = edge_softmax(g, g.edata.pop('att_w'))
        return w

    def gnn(self, g, x):
        g = g.local_var()
        if self._use_KG:
            h = self.entity_embed(g.ndata['id'])
        else:
            h = torch.cat((self.item_proj(x[0]), self.user_proj(x[1])), dim=0)
        node_embed_cache = [h]
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            out = F.normalize(h, p=2, dim=1)
            node_embed_cache.append(out)
        final_h = torch.cat(node_embed_cache, 1)
        return final_h

    def get_loss(self, embedding, src_ids, pos_dst_ids, neg_dst_ids):
        src_vec = embedding[src_ids]
        pos_dst_vec = embedding[pos_dst_ids]
        neg_dst_vec = embedding[neg_dst_ids]
        pos_score = torch.bmm(src_vec.unsqueeze(1), pos_dst_vec.unsqueeze(2)).squeeze()  #
        neg_score = torch.bmm(src_vec.unsqueeze(1), neg_dst_vec.unsqueeze(2)).squeeze()  #
        cf_loss = torch.mean(F.logsigmoid(pos_score - neg_score) ) * (-1.0)
        reg_loss = _L2_loss_mean(src_vec) + _L2_loss_mean(pos_dst_vec) + _L2_loss_mean(neg_dst_vec)
        return cf_loss + self._reg_lambda_gnn * reg_loss


class NegativeSampler(object):
    def __init__(self, g, k):
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for etype in g.canonical_etypes}
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinomial(len(src), replacement=True)
            result_dict[etype] = (src, dst)
        return result_dict
    
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"]

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']
        
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]