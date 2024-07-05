import itertools
import os

os.environ["DGLBACKEND"] = "pytorch"

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve,precision_recall_fscore_support
# from torcheval.metrics.functional import binary_auprc




class metricsTrain():
    def __init__(self, device):
        self.device = device

    def compute_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).view(scores.shape[0], 1).to(self.device)
        return F.binary_cross_entropy_with_logits(scores, labels)
    
    def compute_loss_hetero(self, pos_score, neg_score):
        n_edges = pos_score.shape[0]
        return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()

    def compute_auc(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).numpy()
        return roc_auc_score(labels, scores)
    
    def compute_prc_rec_curve(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).numpy()
        return precision_recall_curve(labels, scores)
    
    def compute_prc_recall_f1(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).numpy()
        return precision_recall_fscore_support(labels, scores, average='binary')
    
    
