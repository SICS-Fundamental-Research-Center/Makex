
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import copy
import random as rd
import os
os.environ["DGLBACKEND"] = "pytorch"

import sys 
from model import ScorePredictor, DotPredictor

__all__ = ["KGATPGExp", "HGTPGExp", "PinsagePGExp"]

        
class KGATPGExp(nn.Module):

    def __init__(
        self,
        model,
        num_features,
        num_hops=None,
        explain_graph=True,
        coff_budget=0.01,
        coff_connect=5e-4,
        sample_bias=0.0,
    ):
        super(KGATPGExp, self).__init__()

        self.model = model
        self.graph_explanation = explain_graph
        self.num_features = num_features * (2 if self.graph_explanation else 4)
        self.num_hops = num_hops
        self.coff_budget = coff_budget
        self.coff_connect = coff_connect
        self.sample_bias = sample_bias

        self.init_bias = 0.0

        self.elayers = nn.Sequential(
            nn.Linear(self.num_features, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def set_masks(self, graph, edge_mask=None):
        if edge_mask is None:
            num_nodes = graph.num_nodes()
            num_edges = graph.num_edges()

            init_bias = self.init_bias
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (2 * num_nodes)
            )
            self.edge_mask = torch.randn(num_edges) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask = self.edge_mask.to(graph.device)

    def clear_masks(self):
        self.edge_mask = None

    def parameters(self):
        return self.elayers.parameters()

    def loss(self, prob, ori_pred):
        target_prob = prob.gather(-1, ori_pred.unsqueeze(-1))
        target_prob += 1e-6
        pred_loss = torch.mean(-torch.log(target_prob))

        edge_mask = self.sparse_mask_values
        if self.coff_budget <= 0:
            size_loss = self.coff_budget * torch.sum(edge_mask)
        else:
            size_loss = self.coff_budget * 0.0001 * F.relu(
                torch.sum(edge_mask) - self.coff_budget
            )

        scale = 0.99
        edge_mask = self.edge_mask * (2 * scale - 1.0) + (1.0 - scale)
        mask_ent = -edge_mask * torch.log(edge_mask) - (
            1 - edge_mask
        ) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_connect * torch.mean(mask_ent)

        print("pred_loss{} size_loss{} mask_ent_loss{}".format(pred_loss, size_loss, mask_ent_loss))
        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def concrete_sample(self, w, beta=1.0, training=True):
        if training:
            bias = self.sample_bias
            random_noise = torch.rand(w.size()).to(w.device)
            random_noise = bias + (1 - 2 * bias) * random_noise
            gate_inputs = torch.log(random_noise) - torch.log(
                1.0 - random_noise
            )
            gate_inputs = (gate_inputs + w) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(w)

        return gate_inputs

    def train_step_link(self, graph, feat, train_pos_g, temperature, **kwargs):
        print(self.graph_explanation)
        self.model = self.model.to(graph.device)
        self.model.eval()
        self.elayers = self.elayers.to(graph.device)

        self.compute_pred_score = DotPredictor()
        with torch.no_grad():
            print("Compute attention weight in eval func ...")
            A_w = self.model.compute_attention(graph)
            graph.edata['w'] = A_w

        traing_edges = train_pos_g.edges(form='all', order='eid')
        train_srcs = traing_edges[0]
        train_dsts = traing_edges[1]
        rd_ids = rd.choices(range(len(train_srcs)), k=2)
        for i in rd_ids:
            print(i)
        sample_srcs = [train_srcs[i] for i in rd_ids]
        sample_dsts = [train_dsts[i] for i in rd_ids]
        print("train srcs : {}, dsts: {}".format(len(train_srcs), len(train_dsts)))
        
        prob, _, batched_graph, inverse_indices = self.explain_link(
            graph, feat, sample_srcs, sample_dsts, temperature, training=True, **kwargs
        )
        pred = self.model.gnn(batched_graph, self.batched_feats)
        self.batched_feats = None
        pred = pred.argmax(-1).data

        loss = self.loss(prob[inverse_indices], pred[inverse_indices])
        return loss

    def explain_link(
        self, graph, feat, srcs, dsts, temperature=1.0, training=False, **kwargs
    ):
        if isinstance(srcs, torch.Tensor):
            srcs = srcs.tolist()
        if isinstance(srcs, int):
            srcs = [srcs]

        if isinstance(dsts, torch.Tensor):
            dsts = dsts.tolist()
        if isinstance(dsts, int):
            dsts = [dsts]

        with torch.no_grad():
            print("Compute attention weight in eval func ...")
            A_w = self.model.compute_attention(graph)
            graph.edata['w'] = A_w

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        batched_graph = []
        batched_embed = []

        for idx in range(len(srcs)):
            src = srcs[idx]
            dst = dsts[idx]
            
            pair = [src, dst]
            sg, inverse_indices = dgl.khop_in_subgraph(
                graph, pair, self.num_hops
            )
            print("sg inverse indices {}".format(inverse_indices))
            sg.ndata["feat"] = feat[sg.ndata[dgl.NID].long()]
            sg.ndata["train"] = torch.tensor(
                [nid in inverse_indices for nid in sg.nodes()], device=sg.device
            )

            embed = self.model.gnn(sg, sg.ndata["feat"])
            embed = embed.data

            col, row = sg.edges()
            col_emb = embed[col.long()]
            row_emb = embed[row.long()]
            self_src_emb = embed[inverse_indices[0]].repeat(sg.num_edges(), 1)
            self_dst_emb = embed[inverse_indices[1]].repeat(sg.num_edges(), 1)
            emb = torch.cat([col_emb, row_emb, self_src_emb, self_dst_emb], dim=-1)
            batched_embed.append(emb)
            batched_graph.append(sg)

        batched_graph = dgl.batch(batched_graph)

        batched_embed = torch.cat(batched_embed)
        batched_embed = self.elayers(batched_embed)
        values = batched_embed.reshape(-1)

        values = self.concrete_sample(
            values, beta=temperature, training=training
        )
        self.sparse_mask_values = values

        col, row = batched_graph.edges()
        edge_mask = values

        self.set_masks(batched_graph, edge_mask)
        print("batch mask shape {}".format(self.edge_mask.shape))

        batched_feats = batched_graph.ndata["feat"]
        h = self.edge_mask.sigmoid()
        eid_to_rm = torch.where(h < 0.5)
        new_graph = copy.deepcopy(batched_graph)
        if len(eid_to_rm[0]) > 0:
            new_graph = dgl.remove_edges(new_graph, eid_to_rm[0], store_ids=True)
        logits = self.model.gnn(new_graph, new_graph.ndata['feat'])
        probs = F.softmax(logits, dim=-1)
        print(probs.shape)

        batched_inverse_indices = (
            batched_graph.ndata["train"].nonzero().squeeze(1)
        )

        if training:
            self.batched_feats = batched_feats
            probs = probs.data
        else:
            self.clear_masks()

        return (
            probs,
            edge_mask,
            batched_graph,
            batched_inverse_indices,
        )

class HGTPGExp(nn.Module):

    def __init__(
        self,
        model,
        num_features,
        num_hops=None,
        explain_graph=True,
        coff_budget=0.01,
        coff_connect=5e-4,
        sample_bias=0.0,
    ):
        super(HGTPGExp, self).__init__()

        self.model = model
        self.graph_explanation = explain_graph
        self.num_features = num_features * (2 if self.graph_explanation else 4)
        self.num_hops = num_hops

        self.coff_budget = coff_budget
        self.coff_connect = coff_connect
        self.sample_bias = sample_bias

        self.init_bias = 0.0

        self.elayers = nn.Sequential(
            nn.Linear(self.num_features, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def set_masks(self, graph, edge_mask=None):
        if edge_mask is None:
            num_nodes = graph.num_nodes()
            num_edges = graph.num_edges()

            init_bias = self.init_bias
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (2 * num_nodes)
            )
            self.edge_mask = torch.randn(num_edges) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask = self.edge_mask.to(graph.device)

    def clear_masks(self):
        self.edge_mask = None

    def parameters(self):
        return self.elayers.parameters()

    def loss(self, prob, ori_pred):
        target_prob = prob.gather(-1, ori_pred.unsqueeze(-1))
        target_prob += 1e-6
        pred_loss = torch.mean(-torch.log(target_prob))

        edge_mask = self.sparse_mask_values
        if self.coff_budget <= 0:
            size_loss = self.coff_budget * torch.sum(edge_mask)
        else:
            size_loss = self.coff_budget * 0.0001 * F.relu(
                torch.sum(edge_mask) - self.coff_budget
            )

        scale = 0.99
        edge_mask = self.edge_mask * (2 * scale - 1.0) + (1.0 - scale)
        mask_ent = -edge_mask * torch.log(edge_mask) - (
            1 - edge_mask
        ) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_connect * torch.mean(mask_ent)

        print("pred_loss{} size_loss{} mask_ent_loss{}".format(pred_loss, size_loss, mask_ent_loss))
        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def concrete_sample(self, w, beta=1.0, training=True):
        if training:
            bias = self.sample_bias
            random_noise = torch.rand(w.size()).to(w.device)
            random_noise = bias + (1 - 2 * bias) * random_noise
            gate_inputs = torch.log(random_noise) - torch.log(
                1.0 - random_noise
            )
            gate_inputs = (gate_inputs + w) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(w)

        return gate_inputs

    def train_step_link(self, graph, feat, train_pos_g, temperature, **kwargs):
        print(self.graph_explanation)

        self.model = self.model.to(graph.device)
        self.model.eval()
        self.elayers = self.elayers.to(graph.device)

        self.compute_pred_score = DotPredictor()

        traing_edges = train_pos_g.edges(form='all', order='eid')
        train_srcs = traing_edges[0]
        train_dsts = traing_edges[1]
        rd_ids = rd.choices(range(len(train_srcs)), k=2)
        sample_srcs = [train_srcs[i] for i in rd_ids]
        sample_dsts = [train_dsts[i] for i in rd_ids]
        print("train srcs : {}, dsts: {}".format(len(train_srcs), len(train_dsts)))
        
        prob, _, batched_graph, inverse_indices = self.explain_link(
            graph, feat, sample_srcs, sample_dsts, temperature, training=True, **kwargs
        )
        
        pred = self.model(batched_graph, self.batched_feats, self.batched_ntypes, self.batched_etypes)
        pred = pred.argmax(-1).data

        loss = self.loss(prob[inverse_indices], pred[inverse_indices])
        return loss

    def explain_link(
        self, graph, feat, srcs, dsts, temperature=1.0, training=False, **kwargs
    ):
        if isinstance(srcs, torch.Tensor):
            srcs = srcs.tolist()
        if isinstance(srcs, int):
            srcs = [srcs]

        if isinstance(dsts, torch.Tensor):
            dsts = dsts.tolist()
        if isinstance(dsts, int):
            dsts = [dsts]

        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        batched_graph = []
        batched_embed = []
        print("{} src, {} dsts".format(len(srcs), len(dsts)))

        for idx in range(len(srcs)):
            src = srcs[idx]
            dst = dsts[idx]
            
            pair = [src, dst]
            sg, inverse_indices = dgl.khop_in_subgraph(
                graph, pair, self.num_hops
            )
            print("sg inverse indices {}".format(inverse_indices))
            sg.ndata["feat"] = feat[sg.ndata[dgl.NID].long()]
            sg.ndata["train"] = torch.tensor(
                [nid in inverse_indices for nid in sg.nodes()], device=sg.device
            )

            embed = self.model(sg, sg.ndata['feat'], sg.ndata['ntype'], sg.edata['type'])
            embed = embed.data

            col, row = sg.edges()
            col_emb = embed[col.long()]
            row_emb = embed[row.long()]
            self_src_emb = embed[inverse_indices[0]].repeat(sg.num_edges(), 1)
            self_dst_emb = embed[inverse_indices[1]].repeat(sg.num_edges(), 1)
            emb = torch.cat([col_emb, row_emb, self_src_emb, self_dst_emb], dim=-1)
            batched_embed.append(emb)
            batched_graph.append(sg)

        batched_graph = dgl.batch(batched_graph)

        batched_embed = torch.cat(batched_embed)
        batched_embed = self.elayers(batched_embed)
        values = batched_embed.reshape(-1)

        values = self.concrete_sample(
            values, beta=temperature, training=training
        )
        self.sparse_mask_values = values

        edge_mask = values

        self.set_masks(batched_graph, edge_mask)
        print("batch mask shape {}".format(self.edge_mask.shape))

        batched_feats = batched_graph.ndata["feat"]
        batched_ntypes = batched_graph.ndata['ntype']
        batched_etypes = batched_graph.edata['type']
        h = self.edge_mask.sigmoid()
        eid_to_rm = torch.where(h < 0.5)
        new_graph = copy.deepcopy(batched_graph)
        if len(eid_to_rm[0]) > 0:
            new_graph = dgl.remove_edges(new_graph, eid_to_rm[0], store_ids=True)
        logits = self.model(new_graph, new_graph.ndata['feat'], new_graph.ndata['ntype'], new_graph.edata['type'])
        probs = F.softmax(logits, dim=-1)
        print(probs.shape)

        batched_inverse_indices = (
            batched_graph.ndata["train"].nonzero().squeeze(1)
        )

        if training:
            self.batched_feats = batched_feats
            self.batched_ntypes = batched_ntypes
            self.batched_etypes = batched_etypes
            probs = probs.data
        else:
            self.clear_masks()

        return (
            probs,
            edge_mask,
            batched_graph,
            batched_inverse_indices,
        )

class PinsagePGExp(nn.Module):

    def __init__(
        self,
        model,
        num_features,
        num_hops=None,
        explain_graph=True,
        coff_budget=0.01,
        coff_connect=5e-4,
        sample_bias=0.0,
    ):
        super(PinsagePGExp, self).__init__()

        self.model = model
        self.graph_explanation = explain_graph
        self.num_features = num_features * (2 if self.graph_explanation else 4)
        self.num_hops = num_hops

        self.coff_budget = coff_budget
        self.coff_connect = coff_connect
        self.sample_bias = sample_bias

        self.init_bias = 0.0

        self.elayers = nn.Sequential(
            nn.Linear(self.num_features, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def set_masks(self, graph, edge_mask=None):
        if edge_mask is None:
            num_nodes = graph.num_nodes()
            num_edges = graph.num_edges()

            init_bias = self.init_bias
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (2 * num_nodes)
            )
            self.edge_mask = torch.randn(num_edges) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask = self.edge_mask.to(graph.device)

    def clear_masks(self):
        self.edge_mask = None

    def parameters(self):
        return self.elayers.parameters()

    def loss(self, prob, ori_pred):
        target_prob = prob.gather(-1, ori_pred.unsqueeze(-1))
        target_prob += 1e-6
        pred_loss = torch.mean(-torch.log(target_prob))

        edge_mask = self.sparse_mask_values
        if self.coff_budget <= 0:
            size_loss = self.coff_budget * torch.sum(edge_mask)
        else:
            size_loss = self.coff_budget * 0.0001 * F.relu(
                torch.sum(edge_mask) - self.coff_budget
            )

        scale = 0.99
        edge_mask = self.edge_mask * (2 * scale - 1.0) + (1.0 - scale)
        mask_ent = -edge_mask * torch.log(edge_mask) - (
            1 - edge_mask
        ) * torch.log(1 - edge_mask)
        mask_ent_loss = self.coff_connect * torch.mean(mask_ent)

        print("pred_loss{} size_loss{} mask_ent_loss{}".format(pred_loss, size_loss, mask_ent_loss))
        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def concrete_sample(self, w, beta=1.0, training=True):
        if training:
            bias = self.sample_bias
            random_noise = torch.rand(w.size()).to(w.device)
            random_noise = bias + (1 - 2 * bias) * random_noise
            gate_inputs = torch.log(random_noise) - torch.log(
                1.0 - random_noise
            )
            gate_inputs = (gate_inputs + w) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(w)

        return gate_inputs
    
    def train_step_link(self, graph, feat, train_pos_g, temperature, **kwargs):
        self.model = self.model.to(graph.device)
        self.elayers = self.elayers.to(graph.device)

        traing_edges = train_pos_g.edges(etype=('user', 'rate', 'item'),form='all', order='eid')
        train_srcs = traing_edges[0]
        train_dsts = traing_edges[1]
        rd_ids = rd.choices(range(len(train_srcs)), k=1)
        for i in rd_ids:
            print(i)
        sample_srcs = [train_srcs[i] for i in rd_ids]
        sample_dsts = [train_dsts[i] for i in rd_ids]
        sample_pair = {
            'user': sample_srcs,
            'item': sample_dsts
        }
        print("train srcs : {}, dsts: {}".format(len(train_srcs), len(train_dsts)))

        prob, _, batched_graph, inverse_indices = self.explain_link(
            graph, feat, sample_pair, temperature, training=True, **kwargs
        )

        input_features = batched_graph.srcdata['feat']
        blocks = [batched_graph, batched_graph]
        pred = self.model(blocks, input_features)
        pred = {ntype: pred[ntype].argmax(-1).data for ntype in pred.keys()}

        loss = self.loss(
            torch.cat(
                [prob[ntype][nid] for ntype, nid in inverse_indices.items()]
            ),
            torch.cat(
                [pred[ntype][nid] for ntype, nid in inverse_indices.items()]
            ),
        )
        return loss

    def explain_link(
            self, graph, feat, pair_dict, temperature=1.0, training=False, **kwargs
        ):
            # assert (
            #     not self.graph_explanation
            # ), '"explain_graph" must be False when initializing the module.'
            # assert (
            #     self.num_hops is not None
            # ), '"num_hops" must be provided when initializing the module.'
            # only one pair for training at each epoch

            self.model = self.model.to(graph.device)
            self.elayers = self.elayers.to(graph.device)

            batched_embed = []
            batched_homo_graph = []
            batched_hetero_graph = []
            sg, inverse_indices = dgl.khop_in_subgraph(
                        graph, pair_dict, self.num_hops
                    )

            for sg_ntype in sg.ntypes:
                sg_feat = feat[sg_ntype][sg.ndata[dgl.NID][sg_ntype].long()]
                train_mask = [
                    sg_ntype in inverse_indices
                    and node_id in inverse_indices[sg_ntype]
                    for node_id in sg.nodes(sg_ntype)
                ]

                sg.nodes[sg_ntype].data["feat"] = sg_feat
                sg.nodes[sg_ntype].data["train"] = torch.tensor(
                    train_mask, device=sg.device
                )

            input_features = sg.srcdata['feat']
            sg_blocks = [sg, sg]
            # h = model(blocks, input_features)
            embed = self.model(sg_blocks, input_features)
            for ntype in embed.keys():
                sg.nodes[ntype].data["emb"] = embed[ntype].data

            homo_sg = dgl.to_homogeneous(sg, ndata=["emb"])
            homo_sg_embed = homo_sg.ndata["emb"]

            col, row = homo_sg.edges()
            col_emb = homo_sg_embed[col.long()]
            row_emb = homo_sg_embed[row.long()]
            self_src_emb = homo_sg_embed[
                inverse_indices['user'][0]
            ].repeat(sg.num_edges(), 1)
            self_dst_emb = homo_sg_embed[
                inverse_indices['item'][0]
            ].repeat(sg.num_edges(), 1)
            emb = torch.cat([col_emb, row_emb, self_src_emb, self_dst_emb], dim=-1)
            batched_embed.append(emb)
            batched_homo_graph.append(homo_sg)
            batched_hetero_graph.append(sg)
            # for target_ntype, target_nids in nodes.items():
            #     if isinstance(target_nids, torch.Tensor):
            #         target_nids = target_nids.tolist()

            #     for target_nid in target_nids:
            #         sg, inverse_indices = dgl.khop_in_subgraph(
            #             graph, {target_ntype: target_nid}, self.num_hops
            #         )

            #         for sg_ntype in sg.ntypes:
            #             sg_feat = feat[sg_ntype][sg.ndata[dgl.NID][sg_ntype].long()]
            #             train_mask = [
            #                 sg_ntype in inverse_indices
            #                 and node_id in inverse_indices[sg_ntype]
            #                 for node_id in sg.nodes(sg_ntype)
            #             ]

            #             sg.nodes[sg_ntype].data["feat"] = sg_feat
            #             sg.nodes[sg_ntype].data["train"] = torch.tensor(
            #                 train_mask, device=sg.device
            #             )

            #         embed = self.model(sg, sg.ndata["feat"], embed=True, **kwargs)
            #         for ntype in embed.keys():
            #             sg.nodes[ntype].data["emb"] = embed[ntype].data

            #         homo_sg = dgl.to_homogeneous(sg, ndata=["emb"])
            #         homo_sg_embed = homo_sg.ndata["emb"]

            #         col, row = homo_sg.edges()
            #         col_emb = homo_sg_embed[col.long()]
            #         row_emb = homo_sg_embed[row.long()]
            #         self_emb = homo_sg_embed[
            #             inverse_indices[target_ntype][0]
            #         ].repeat(sg.num_edges(), 1)
            #         emb = torch.cat([col_emb, row_emb, self_emb], dim=-1)
            #         batched_embed.append(emb)
            #         batched_homo_graph.append(homo_sg)
            #         batched_hetero_graph.append(sg)

            batched_homo_graph = dgl.batch(batched_homo_graph)
            batched_hetero_graph = dgl.batch(batched_hetero_graph)

            batched_embed = torch.cat(batched_embed)
            batched_embed = self.elayers(batched_embed)
            values = batched_embed.reshape(-1)

            values = self.concrete_sample(
                values, beta=temperature, training=training
            )
            self.sparse_mask_values = values

            # col, row = batched_homo_graph.edges()
            # reverse_eids = batched_homo_graph.edge_ids(row, col).long()
            # edge_mask = (values + values[reverse_eids]) / 2
            edge_mask = values

            self.set_masks(batched_homo_graph, edge_mask)

            # Convert the edge mask back into heterogeneous format.
            hetero_edge_mask = self._edge_mask_to_heterogeneous(
                edge_mask=edge_mask,
                homograph=batched_homo_graph,
                heterograph=batched_hetero_graph,
            )

            batched_feats = {
                ntype: batched_hetero_graph.nodes[ntype].data["feat"]
                for ntype in batched_hetero_graph.ntypes
            }

            new_graph = copy.deepcopy(batched_hetero_graph)
            for canonical_etype, canonical_etype_mask in hetero_edge_mask.items():
                temp_edge_mask= canonical_etype_mask.sigmoid()
                # print(temp_edge_mask)
                eid_to_rm = torch.where(temp_edge_mask < 0.5)
                
                # print(eid_to_rm)
                if len(eid_to_rm[0]) > 0:
                    # print("here")
                    new_graph = dgl.remove_edges(new_graph, eid_to_rm[0], canonical_etype, store_ids=True)
            new_blocks = [new_graph, new_graph]
            # The model prediction with the updated edge mask.
            logits = self.model(
                new_blocks,
                batched_feats
            )

            probs = {
                ntype: F.softmax(logits[ntype], dim=-1) for ntype in logits.keys()
            }

            batched_inverse_indices = {
                ntype: batched_hetero_graph.nodes[ntype]
                .data["train"]
                .nonzero()
                .squeeze(1)
                for ntype in batched_hetero_graph.ntypes
            }

            if training:
                self.batched_feats = batched_feats
                probs = {ntype: probs[ntype].data for ntype in probs.keys()}
            else:
                self.clear_masks()

            return (
                probs,
                hetero_edge_mask,
                batched_hetero_graph,
                batched_inverse_indices,
            )

    def _edge_mask_to_heterogeneous(self, edge_mask, homograph, heterograph):
        return {
            etype: edge_mask[
                (homograph.edata[dgl.ETYPE] == heterograph.get_etype_id(etype))
                .nonzero()
                .squeeze(1)
            ]
            for etype in heterograph.canonical_etypes
        }
