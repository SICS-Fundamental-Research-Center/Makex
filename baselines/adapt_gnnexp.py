from math import sqrt

import torch
import dgl
from torch import nn
from tqdm import tqdm
import copy

import os
import sys 
sys.path.append(".")

os.environ["DGLBACKEND"] = "pytorch"
from model import ScorePredictor

__all__ = ["KGATExp", "HGTExp", "PinsageExp"]

class HGTExp(nn.Module):

    def __init__(
        self,
        model,
        num_hops,
        lr=0.01,
        num_epochs=100,
        *,
        alpha1=0.005,
        alpha2=1.0,
        beta1=1.0,
        beta2=0.1,
        log=True,
    ):
        super(HGTExp, self).__init__()
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.log = log

    def _init_masks(self, graph, feat):
        
        num_nodes, feat_size = feat.size()
        num_edges = graph.num_edges()
        device = feat.device

        std = 0.1
        feat_mask = nn.Parameter(torch.randn(1, feat_size, device=device) * std)

        std = nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * num_nodes))
        edge_mask = nn.Parameter(torch.randn(num_edges, device=device) * std)

        return feat_mask, edge_mask

    def _loss_regularize(self, loss, feat_mask, edge_mask):
        
        eps = 1e-15

        edge_mask = edge_mask.sigmoid()
        loss = loss + self.alpha1 * torch.sum(edge_mask)
        ent = -edge_mask * torch.log(edge_mask + eps) - (
            1 - edge_mask
        ) * torch.log(1 - edge_mask + eps)
        loss = loss + self.alpha2 * ent.mean()

        feat_mask = feat_mask.sigmoid()
        loss = loss + self.beta1 * torch.mean(feat_mask)
        ent = -feat_mask * torch.log(feat_mask + eps) - (
            1 - feat_mask
        ) * torch.log(1 - feat_mask + eps)
        loss = loss + self.beta2 * ent.mean()

        return loss

    def explain_link(self, graph, feat, x, y, **kwargs):
        self.model = self.model.to(graph.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(graph, graph.ndata['feat'], graph.ndata['ntype'], graph.edata['type'])
            pred_label = torch.dot(logits[x], logits[y])

        feat_mask, edge_mask = self._init_masks(graph, feat)

        params = [feat_mask, edge_mask]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description("Explain")

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            new_feat = feat * feat_mask.sigmoid()
            h = edge_mask.sigmoid()
            eid_to_rm = torch.where(h < 0.5)
            new_graph = copy.deepcopy(graph)
            new_graph.ndata['feat'] = new_feat
            if len(eid_to_rm[0]) > 0:
                new_graph = dgl.remove_edges(new_graph, eid_to_rm[0], store_ids=True)
            logits = self.model(new_graph, new_graph.ndata['feat'], new_graph.ndata['ntype'], new_graph.edata['type'])
            logit_probs = torch.dot(logits[x], logits[y])
            loss = (logit_probs - pred_label)
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        feat_mask = feat_mask.detach().sigmoid().squeeze()
        edge_mask = edge_mask.detach().sigmoid()

        return feat_mask, edge_mask
    
class KGATExp(nn.Module):

    def __init__(
        self,
        model,
        num_hops,
        lr=0.01,
        num_epochs=100,
        *,
        alpha1=0.005,
        alpha2=1.0,
        beta1=1.0,
        beta2=0.1,
        log=True,
    ):
        super(KGATExp, self).__init__()
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.log = log

    def _init_masks(self, graph, feat):
        
        num_nodes, feat_size = feat.size()
        num_edges = graph.num_edges()
        device = feat.device

        std = 0.1
        feat_mask = nn.Parameter(torch.randn(1, feat_size, device=device) * std)

        std = nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * num_nodes))
        edge_mask = nn.Parameter(torch.randn(num_edges, device=device) * std)
        return feat_mask, edge_mask

    def _loss_regularize(self, loss, feat_mask, edge_mask):
        eps = 1e-15

        edge_mask = edge_mask.sigmoid()
        loss = loss + self.alpha1 * torch.sum(edge_mask)
        ent = -edge_mask * torch.log(edge_mask + eps) - (
            1 - edge_mask
        ) * torch.log(1 - edge_mask + eps)
        loss = loss + self.alpha2 * ent.mean()

        feat_mask = feat_mask.sigmoid()
        loss = loss + self.beta1 * torch.mean(feat_mask)
        ent = -feat_mask * torch.log(feat_mask + eps) - (
            1 - feat_mask
        ) * torch.log(1 - feat_mask + eps)
        loss = loss + self.beta2 * ent.mean()

        return loss

    def explain_link(self, graph, feat, x, y, **kwargs):
        self.model = self.model.to(graph.device)
        self.model.eval()

        with torch.no_grad():
            print("Compute attention weight in eval func ...")
            A_w = self.model.compute_attention(graph)
            graph.edata['w'] = A_w
            logits = self.model.gnn(graph, graph.ndata['feat'])
            pred_label = torch.dot(logits[x], logits[y])

        feat_mask, edge_mask = self._init_masks(graph, feat)

        params = [feat_mask, edge_mask]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        if self.log:
            pbar = tqdm(total=self.num_epochs)
            pbar.set_description("Explain")

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            new_feat = feat * feat_mask.sigmoid()
            h = edge_mask.sigmoid()
            eid_to_rm = torch.where(h < 0.5)
            new_graph = copy.deepcopy(graph)
            new_graph.ndata['feat'] = new_feat
            if len(eid_to_rm[0]) > 0:
                new_graph = dgl.remove_edges(new_graph, eid_to_rm[0], store_ids=True)
            logits = self.model.gnn(new_graph, new_graph.ndata['feat'])
            logit_probs = torch.dot(logits[x], logits[y])
            loss = (logit_probs - pred_label)
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward()
            optimizer.step()

            if self.log:
                pbar.update(1)

        if self.log:
            pbar.close()

        feat_mask = feat_mask.detach().sigmoid().squeeze()
        edge_mask = edge_mask.detach().sigmoid()

        return feat_mask, edge_mask
    

class PinsageExp(nn.Module):

    def __init__(
        self,
        model,
        num_hops,
        lr=0.01,
        num_epochs=100,
        *,
        alpha1=0.005,
        alpha2=1.0,
        beta1=1.0,
        beta2=0.1,
        log=True,
    ):
        super(PinsageExp, self).__init__()
        self.model = model
        self.num_hops = num_hops
        self.lr = lr
        self.num_epochs = num_epochs
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.log = log

    def _init_masks(self, graph, feat):
        device = graph.device
        feat_masks = {}
        std = 0.1
        for node_type, feature in feat.items():
            _, feat_size = feature.size()
            feat_masks[node_type] = nn.Parameter(
                torch.randn(1, feat_size, device=device) * std
            )

        edge_masks = {}
        for canonical_etype in graph.canonical_etypes:
            src_num_nodes = graph.num_nodes(canonical_etype[0])
            dst_num_nodes = graph.num_nodes(canonical_etype[-1])
            num_nodes_sum = src_num_nodes + dst_num_nodes
            num_edges = graph.num_edges(canonical_etype)
            std = nn.init.calculate_gain("relu")
            if num_nodes_sum > 0:
                std *= sqrt(2.0 / num_nodes_sum)
            edge_masks[canonical_etype] = nn.Parameter(
                torch.randn(num_edges, device=device) * std
            )

        return feat_masks, edge_masks

    def _loss_regularize(self, loss, feat_masks, edge_masks):
        eps = 1e-15

        for edge_mask in edge_masks.values():
            edge_mask = edge_mask.sigmoid()
            loss = loss + self.alpha1 * torch.sum(edge_mask)
            ent = -edge_mask * torch.log(edge_mask + eps) - (
                1 - edge_mask
            ) * torch.log(1 - edge_mask + eps)
            loss = loss + self.alpha2 * ent.mean()

        for feat_mask in feat_masks.values():
            feat_mask = feat_mask.sigmoid()
            loss = loss + self.beta1 * torch.mean(feat_mask)
            ent = -feat_mask * torch.log(feat_mask + eps) - (
                1 - feat_mask
            ) * torch.log(1 - feat_mask + eps)
            loss = loss + self.beta2 * ent.mean()

        return loss

    def explain_link(self, graph, feat, x, y, **kwargs):
        self.model = self.model.to(graph.device)
        self.model.eval()

        blocks = [graph, graph]
        with torch.no_grad():
            logits = self.model(blocks, feat)
            
            pred_label = torch.dot(logits['user'][x], logits['item'][y])
            print(pred_label)

        feat_mask, edge_mask = self._init_masks(graph, feat)

        params = [*feat_mask.values(), *edge_mask.values()]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            h = {}
            for node_type, node_feat in feat.items():
                h[node_type] = node_feat * feat_mask[node_type].sigmoid()
            eweight = {}
            new_graph = copy.deepcopy(graph)
            for canonical_etype, canonical_etype_mask in edge_mask.items():
                temp_edge_mask= canonical_etype_mask.sigmoid()
                print(temp_edge_mask)
                eid_to_rm = torch.where(temp_edge_mask < 0.5)
                if len(eid_to_rm[0]) > 0:
                    print("here")
                    new_graph = dgl.remove_edges(new_graph, eid_to_rm[0], canonical_etype, store_ids=True)
            
            new_blocks = [new_graph, new_graph]
            logits = self.model(new_blocks, h)
            log_probs = torch.dot(logits['user'][x], logits['item'][y])
            loss = (log_probs - pred_label)
            loss = self._loss_regularize(loss, feat_mask, edge_mask)
            loss.backward(retain_graph=True)
            optimizer.step()

        for node_type in feat_mask:
            feat_mask[node_type] = (
                feat_mask[node_type].detach().sigmoid().squeeze()
            )

        for canonical_etype in edge_mask:
            edge_mask[canonical_etype] = (
                edge_mask[canonical_etype].detach().sigmoid()
            )

        return feat_mask, edge_mask
