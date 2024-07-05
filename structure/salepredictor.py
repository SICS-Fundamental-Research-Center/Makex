from torch.utils import data
from structure.topkheap import TopKHeap
import pyMakex
import sys
import multiprocessing
import copy
import time
import os
sys.path.append("..")
import tool.Match as Match


class SalePredictor:
    def __init__(self, top_k, sort_criteria, num_process=1) -> None:
        self.final_rep_list = []
        self.top_k_result = TopKHeap(top_k)
        self.num_process = num_process
        self.tot_supp_R = set()
        self.tot_supp_Q = set()
        self.sort_criteria = sort_criteria

    def REPListArg(self, rep_list):
        res = []
        for single_rep in rep_list:
            res.append(single_rep.GetREPMatchArg())
        return res

    def REPScore(self, rep, data_graph):
        supp = Match.REPMatch(rep, data_graph, use_ptime=True, save_result=False)

        supp_Q = supp[1]
        supp_R = supp[0]
        positive_pair_num = data_graph.PositivePairNum()

        if supp_Q == 0 or positive_pair_num == 0 or supp_R == 0:
            return (0, [], [])
        recall = supp_R / positive_pair_num
        conf = supp_R / supp_Q
        if self.sort_criteria == "conf":
            return [conf, supp[0], supp[1]]
        elif self.sort_criteria == "f1":
            return [2 * conf * recall / (conf + recall), supp[0], supp[1]]

    def ScoreFunction(self, rep, data_graph, mp_lock=None, return_rep_flag=False):
        rep_score, supp_R, supp_Q = self.REPScore(rep, data_graph)
        if return_rep_flag == False:
            return rep_score
        else:
            return (rep_score, rep, supp_R, supp_Q)

    def ScoreFunction2(self, rep, data_graph, return_rep_flag=False, ele_idx = 0):
        rep_score, supp_R, supp_Q = self.REPScore(rep, data_graph)
        if return_rep_flag == False:
            return rep_score
        else:
            return (rep_score, rep, supp_R, supp_Q, str(ele_idx) + "finished")

    def ScoreFunction_U_V(self, rep, data_graph, user_id, item_id, return_rep_flag=False, ele_idx = 0, enable_topk=1, topk_explanation_scores_list = [], topk = 1):

        if enable_topk == 0 or enable_topk == 2:
            rep_score, supp_R, supp_Q, match_flag = self.REPScore_U_V(rep, data_graph, user_id, item_id, enable_topk, topk_explanation_scores_list, topk, ele_idx)
            if return_rep_flag == False:
                return rep_score
            else:
                return (rep_score, rep, supp_R, supp_Q, str(ele_idx) + "finished", match_flag)
        else:
            topk_explanation, match_flag = self.REPScore_U_V(rep, data_graph, user_id, item_id, enable_topk, topk_explanation_scores_list, topk, ele_idx)
            return (topk_explanation, match_flag)
            

    def REPScore_U_V(self, rep, data_graph, user_id, item_id, enable_topk = 1, topk_explanation_scores_list = [], topk = 1, rep_id = 0):
        save_result = False
        
        supp = Match.REPMatch_U_V(rep, data_graph, user_id, item_id, use_ptime=True, save_result=save_result, enable_topk=enable_topk, topk_explanation_scores_list = topk_explanation_scores_list, topk = topk, rep_id = rep_id)

        e_ = time.time()
        match_flag = 0
        if enable_topk == 1:
            match_flag = supp[3]
            topk_explanation = supp[2]
            return topk_explanation, match_flag
        
        if enable_topk == 0 or enable_topk == 2:
            match_flag = supp[2]
        
        supp_Q = supp[1]
        supp_R = supp[0]
        positive_pair_num = data_graph.PositivePairNum()
        
        if supp_Q == 0 or positive_pair_num == 0 or supp_R == 0:
            return (0, [], [], match_flag)
        recall = supp_R / positive_pair_num
        conf = supp_R / supp_Q

        if self.sort_criteria == "conf":
            return [conf, supp[0], supp[1], match_flag]
        elif self.sort_criteria == "f1":
            return [2 * conf * recall / (conf + recall), supp[0], supp[1], match_flag]


    def SelectHighQualityREP(self, rep_list, data_graph, supp_limit, conf_limit, top_k):
        score_rep_list = []
        l = multiprocessing.Manager().Lock()
        pool = multiprocessing.Pool(processes=self.num_process)
        result = pool.starmap_async(self.ScoreFunction, [(ele, data_graph, l, True) for ele in rep_list]).get()
        pool.close()
        pool.join()

        for ele in result:
            if isinstance(ele[2],list):
                supp_R = len(ele[2])
                supp_Q = len(ele[3])
            else:
                supp_R = ele[2]
                supp_Q = ele[3]

            conf = 0
            if supp_Q != 0:
                conf = supp_R / supp_Q
            if conf >= conf_limit and supp_R >= supp_limit:
                score_rep_list.append(ele)
        def cmp(rep_with_score):
            return rep_with_score[0]

        score_rep_list.sort(key=cmp, reverse=True)
        score_rep_list.reverse()
        self.final_rep_list += [(ele[0], ele[1]) for ele in score_rep_list]
        
        return self.final_rep_list


    def ScoreFunction_makex(self, rep, data_graph, return_rep_flag=False, ele_idx = 0):
        rep_score, supp_R, supp_Q = self.REPScore_makex(rep, data_graph)
        if return_rep_flag == False:
            return rep_score
        else:
            return (rep_score, rep, supp_R, supp_Q, str(ele_idx) + "finished")

    def REPScore_makex(self, rep, data_graph):
        supp = Match.REPMatch(rep, data_graph, use_ptime=True, save_result=False)

        supp_Q = supp[1]
        supp_R = supp[0]
        positive_pair_num = data_graph.PositivePairNum()
        

        if supp_Q == 0 or positive_pair_num == 0 or supp_R == 0:
            return (0, [], [])
        recall = supp_R / positive_pair_num
        conf = supp_R / supp_Q
        
        if self.sort_criteria == "conf":
            return [conf, supp[0], supp[1]]
        elif self.sort_criteria == "f1":
            return [2 * conf * recall / (conf + recall), supp[0], supp[1]]
