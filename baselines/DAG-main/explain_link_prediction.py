import os
import re
from pprint import pprint
import numpy as np
import json
import random
# import math

import torch
import torch.nn as nn
# import models # this is for old GNN by GNNExplainer
# from dig_models import *
from torch_geometric.data import Data
import torch_geometric
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import pandas as pd

class DAG:
    def __init__(
            self,
            dataset,
            modelname,
            lambdas
    ):
        self.dataset = dataset
        if self.dataset=='movie':
            self.num_pairs = 795334
        elif self.dataset == 'yelp':
            self.num_pairs = 94689
        elif self.dataset=='ciao':
            self.num_pairs=51957
        self.modelname = modelname

        self.gSpan_output_file = 'data/'+self.dataset+'/gSpan_output.txt'
        self.gnn_score_file = 'data/'+self.dataset+'/'+self.modelname+'/gspan/eval.csv'

        # 
        self.true_label = self.getTrueLabel() #
        self.n_total_inst = len(self.true_label[0] + self.true_label[1]) # 
        self.where =self.getWhere() # 
        self.n_subgraph = len(self.where)
        self.score, self.size_dict = self.getGNNscore() # 
        self.diver_dic =self.getDiverDic() # 
        # self.Lambda = Lambda
        # self.Lambda = np.array([lambdas[0], lambdas[1],lambdas[2], 0, 0, 0, lambdas[-1]],dtype=np.int64)
        self.Lambda = np.array([lambdas[0], lambdas[1],lambdas[2], 0, 0, 0, lambdas[-1]])
        self.candidate = self.getCandidate()  # 
        self.distribution = self.getDistribution() # 

        self.n_class = len(self.true_label)
        self.size_D = self.n_subgraph*self.n_class
        self.n_inst_clss = [len(x) for x in self.true_label]
        self.save_path = self.getSavePath()
        print('explainer set up.')

    def getTrueLabel(self):
        # only for true labels
        true_ids =[]
        for i in range(self.num_pairs):
            true_ids.append(i)
        
        false_ids = []
        return [false_ids, true_ids]
    
        # label_file = 'data/'+self.dataset+'/raw/'+self.dataset+'_graph_labels.txt'
        # with open(label_file,'r') as f:
        #     content = f.readlines()
        # if 'MUTAG' in self.dataset:
        #     return [[i for i in range(len(content)) if content[i].strip()=='-1'], [i for i in range(len(content)) if content[i].strip()=='1']]
        # else:
        #     return [[i for i in range(len(content)) if content[i].strip()=='0'], [i for i in range(len(content)) if content[i].strip()=='1']]

    def getWhere(self):
        with open(self.gSpan_output_file, 'r') as f:
            content = f.read()
        where = []
        for w in re.findall(r"where\:\ \[(.*?)\]", content):
            tmp = w.split(', ')
            where.append([int(x) for x in tmp])
        
        print("we print where here")
        # print(where)
        return  where

    def getGNNscore(self):
        if self.dataset=='highschool':

            # gSpan_output_data = TUDataset('data/highschool/', name='gSpan_output_data',use_edge_attr = True)
            # score_loader = DataLoader(gSpan_output_data, batch_size=len(gSpan_output_data), shuffle=False)

            # model.eval()
            # for d in score_loader:
            #     output=model(d).data
            # score = [[x.item() for x in nn.Softmax(dim=0)(o)] for o in output]
            # return score, None

            with open(self.gSpan_output_file,'r') as f:
                content = f.readlines()
            graphs = []
            for line in content[:-1]:
                tmp = line.strip().split()
                if len(tmp)==0:
                    continue
                elif tmp[0]=='t':
                    graphs.append(nx.Graph(id=int(tmp[-1])))
                elif tmp[0]== 'v':
                    graphs[-1].add_nodes_from([(int(tmp[1]), {"label": int(tmp[-1])})])
                elif tmp[0]=='e':
                    graphs[-1].add_edges_from([(int(tmp[1]),int(tmp[2]), {"label": int(tmp[-1])})])
                else:
                    continue
            data_items = []
            base = [0 for i in range(self.input_dim)]
            size = {}
            for i in range(len(graphs)):
                size[i] = {}
                size[i]['n'] = graphs[i].number_of_nodes()
                size[i]['e'] = graphs[i].number_of_edges()
                # data = torch_geometric.utils.convert.from_networkx(graphs[i])
                # x = []
                # for node in graphs[i].nodes:
                #     tmp = base.copy()
                #     tmp[graphs[i].nodes[node]['label']] = 1  # one-hot encoding
                #     x.append(tmp)
                # data.x = torch.tensor(x, dtype=torch.float32)
                # edge_attr = []
                # for edge in graphs[i].edges:
                #     edge_attr.append([graphs[i].edges[edge]['label']])
                #     edge_attr.append([graphs[i].edges[edge]['label']])
                # data.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
                # data_items.append(data)

            gSpanOutput_data = TUDataset('data/', 'sampled_subgraph_s5_l3_u7',   use_edge_attr = True)
            gSpanOutput_dataloader = DataLoader(gSpanOutput_data, batch_size=len(gSpanOutput_data), shuffle=False)

            gSpanOutput_pred_probs = []
            self.model.eval()
            for d in gSpanOutput_dataloader:
                output=self.model(d).data
            score = [[x.item() for x in nn.Softmax(dim=0)(o)] for o in output]

            print('highschool GNN score got from the model')
            return score, size
        else:
            with open(self.gSpan_output_file,'r') as f:
                content = f.readlines()
            graphs = []
            for line in content:
                tmp = line.strip().split()
                if len(tmp)==0:
                    continue
                elif tmp[0]=='t':
                    graphs.append(nx.Graph()) #构造一个graph 存起来
                elif tmp[0]== 'v':
                    graphs[-1].add_nodes_from([(int(tmp[1]), {"label": int(tmp[-1])})])
                elif tmp[0]=='e':
                    graphs[-1].add_edges_from([(int(tmp[1]),int(tmp[2]))])
                else:
                    continue

            # data_items = []
            # base = [0 for i in range(self.input_dim)] #这个是干嘛用的呢？
            size = {}
            for i in range(len(graphs)):
                size[i] = {}
                size[i]['n'] = graphs[i].number_of_nodes()
                size[i]['e'] = graphs[i].number_of_edges()
                # data = torch_geometric.utils.convert.from_networkx(graphs[i])
                # x = []
                # for node in graphs[i].nodes:
                #     tmp = base.copy()
                #     tmp[graphs[i].nodes[node]['label']]=1 # one-hot encoding
                #     x.append(tmp)
                # data.x = torch.tensor(x, dtype=torch.float32)
                # data_items.append(data)

            # gSpanOutput_dataloader = DataLoader(data_items, batch_size=len(data_items), shuffle=False)

            # gSpanOutput_pred_probs = []
            # # gSpanOutput_predictions = []
            # self.model.eval()
            # with torch.no_grad():
            #     for batch in gSpanOutput_dataloader:
            #         logits= self.model(data=batch)
            #         # _, gSpanOutput_predictions = torch.max(logits, -1)
            #         gSpanOutput_pred_probs = nn.Softmax(dim=1)(logits)
            # # gSpanOutput_predictions = gSpanOutput_predictions.tolist()
            # score = gSpanOutput_pred_probs.tolist()
            print('reading GNN scores from file')
            gnn_score_df = pd.read_csv(self.gnn_score_file, names= ['pair_id','topk','rec'])
            score = gnn_score_df['rec'].to_numpy().tolist()
            print(score)
            final_score = []
            for i in score:
                final_score.append([0,i])
            score = final_score
            print('GNN score got')
            print(score)
            return score, size

    def getDistribution(self):

        distribution = np.zeros([self.n_total_inst, self.n_subgraph], dtype=np.int8)

        for i in range(self.n_subgraph):
            for pos in self.where[i]:
                distribution[pos][i]=1

        # distribution = np.zeros([self.n_total_inst, len(self.candidate)], dtype=np.int8)

        # for i in range(len(self.candidate)):
        #     for pos in self.where[self.candidate[i][0]]:
        #         distribution[pos][i]=1

        print('shape of distribution matrix: ('+str(distribution.shape[0])+', '+str(distribution.shape[1])+')')
        print('dtyp of distribution '+str(distribution.dtype))

        return distribution

    def getDiverDic(self):
        with open(self.gSpan_output_file, 'r') as f:
            content = f.read()
        chunks = re.findall(r"t #(.*?)Support", content, flags=re.S)
        graphs_weight = {}
        for item in chunks:
            graph = item.strip().split('\n')
            # print('working on pattern # '+graph[0])
            v_labl = set([int(x.split(' ')[-1]) for x in graph if x[0]=='v'])
            e_labl = set([int(x.split(' ')[-1]) for x in graph if x[0]=='e'])
            graphs_weight[int(graph[0])]=len(v_labl) + len(e_labl)
        return graphs_weight

    def getSavePath(self):
        if not os.path.isdir('result'):
            os.mkdir('result')
        if not os.path.isdir(os.path.join('result', self.dataset)):
            os.mkdir(os.path.join('result', self.dataset))

        save_path = 'result/'+self.dataset+'/'+self.modelname+'/'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        if self.dataset=='isAcyclic':
            save_path = 'result/'+self.dataset+'/'+self.model._get_name()+'_result/'+str(self.isAcyclic_n_nodes)+'_nodes/'
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
        return save_path           

    def calMarginE(self, new, E, detail=False):
        if new==(-1,-1):
            return 0

        current_distr = [ np.sum(self.distribution[:, [x[0] for x in E if x[1]==0]], axis=1), np.sum(self.distribution[:, [x[0] for x in E if x[1]==1]], axis=1)]

        # current_cover_v = np.sum(distribution[:,[x[0] for x in E]], axis=1)

        new_v = self.distribution[:,new[0]]
        # print(new)
        new_score = self.score[new[0]][new[1]]

        support_idx = list(self.true_label[new[1]])
        denail_idx = list(self.true_label[1-new[1]]) # binary classification only

        marginal_gain = []

        # Expresiveness
        if len(E)==0:
            current_f1 = 0
        else:
            current_f1 = sum([self.score[x[0]][x[1]] for x in E])/len(E)
        marginal_gain.append((current_f1*len(E)+new_score)/(len(E)+1) - current_f1)

        # Support
        marginal_gain.append((np.count_nonzero((current_distr[new[1]]+new_v)[support_idx])-np.count_nonzero(current_distr[new[1]][support_idx]))/self.n_total_inst)

        # Denial
        marginal_gain.append(-np.count_nonzero(new_v[denail_idx])/(self.n_total_inst*self.n_subgraph))

        # In-class co-occurrence
        marginal_gain.append(-np.sum(current_distr[new[1]][new_v.nonzero()[0]])/(self.n_total_inst*self.n_subgraph*self.n_subgraph))
        # marginal_gain.append(-np.sum((current_distr[0]+current_distr[1])[support_idx][new_v[support_idx].nonzero()[0]])/(n_total_inst*n_subgraph*n_subgraph))

        # Cross-class co-occurrence
        marginal_gain.append(-np.sum(current_distr[1-new[1]][new_v.nonzero()[0]])/(self.n_total_inst*self.n_subgraph*self.n_subgraph))

        # marginal_gain.append(-np.sum((current_distr[0]+current_distr[1])[denail_idx][new_v[denail_idx].nonzero()[0]])/(n_total_inst*n_subgraph*n_subgraph))

        # Comprehensiveness
        if new[1] not in [x[1] for x in E]:
            marginal_gain.append(1/self.n_class)
        else:
            marginal_gain.append(0)

        # Size
        marginal_gain.append(-1/self.size_D)

        if detail:
            return np.round(np.multiply(self.Lambda, marginal_gain), decimals=5)
        else:
            return np.dot(self.Lambda, np.array(marginal_gain))

    def evalStat(self, E):
        evaluation = []

        # Size
        evaluation.append(len(E))

        # Recognition:
        evaluation.append(sum([self.score[x[0]][x[1]] for x in E])/len(E))

        # Self metric
        support_per = 0
        denial_per = 0
        for e in E:
            _, support, denial = self.evalIndividualExp(e)
            support_per += support/(support+denial)
            denial_per += denial/(support+denial)
        support_per = support_per/len(E)
        denial_per = denial_per/len(E)

        evaluation.append(support_per)
        evaluation.append(denial_per)

        distr_class0 = np.sum(self.distribution[:, [x[0] for x in E if x[1]==0]], axis=1)
        distr_class1 = np.sum(self.distribution[:, [x[0] for x in E if x[1]==1]], axis=1)

        # Coverage
        union_support_0 = np.count_nonzero(distr_class0[list(self.true_label[0])])
        union_support_1 = np.count_nonzero(distr_class1[list(self.true_label[1])])
        evaluation.append((union_support_0+union_support_1)/self.n_total_inst)


        # Disagreement with data
        avg_denial_r_0 = np.sum(distr_class0[list(self.true_label[1])])/self.n_inst_clss[1]
        avg_denial_r_1 = np.sum(distr_class1[list(self.true_label[0])])/self.n_inst_clss[0]

        # n_0 = len([x[0] for x in E if x[1]==0])
        # n_1 = len([x[0] for x in E if x[1]==1])
        evaluation.append((avg_denial_r_0+avg_denial_r_1)/len(E))

        # Redundancy and inconsistency
        distr = distr_class0+distr_class1
        pos = np.argwhere(distr > 1)
        if pos.size>0:
            co = 0
            for p in pos:
                # co += math.comb(distr[p][0],2)
                co +=(distr[p][0]*(distr[p][0]-1))/2
            avg_co_quot = 2*co/(self.n_total_inst*len(E)*(len(E)-1))
        else:
            avg_co_quot = 0
        evaluation.append(avg_co_quot)

        # Comprehensiveness
        return evaluation

    def evalOutput(self, output, read_out = False):
        if len(output)==0:
            print('The output is an empty set')
            return []


        ind_eval = []
        for e in output:
            score, support, denial, edges = self.evalIndividualExp(e)
            ind_eval.append([score, support, denial, edges])
        # print(ind_eval)

        evaluation = np.mean(np.asarray(ind_eval), axis=0).tolist()

        evaluation.append(len(output))
        # distr = np.sum(self.distribution[:, [x[0] for x in output if x[1]==output_class]], axis=1)

        # # Coverage
        # union_support = np.count_nonzero(distr[list(self.true_label[output_class])])
        # evaluation.append(union_support/self.n_inst_clss[output_class])

        # # Disagreement with data
        # avg_denial_r = np.sum(distr[list(self.true_label[1-output_class])])/self.n_inst_clss[1-output_class]
        # evaluation.append(avg_denial_r/len(output))

        # # Redundancy and inconsistency
        # pos = np.argwhere(distr > 1)
        # if pos.size>0:
        #     co = 0
        #     for p in pos:
        #         # co += math.comb(distr[p][0],2)
        #         co +=(distr[p][0]*(distr[p][0]-1))/2
        #     avg_co_quot = 2*co/(self.n_total_inst*len(output)*(len(output)-1))
        # else:
        #     avg_co_quot = 0
        # evaluation.append(avg_co_quot)

        if read_out:
            print('GNN score: '+str(evaluation[0]))
            print('self-sup: '+str(evaluation[1]))
            print('self-den: '+str(evaluation[2]))
            print(f"No.edges: {evaluation[3]}")
            print(f"No.explanantions: {evaluation[-1]}\n")

        target_class= output[0][-1]
        with open(self.save_path+'class_'+str(target_class)+'_fianl_single_quantitatives.json','w') as f:
          json.dump(evaluation,f)

        return evaluation


    def oneMore(self, E, candidate, c):
        tmp = [x for x in candidate if x[1]==c]
        best_gain = -10000
        for j in range(len(tmp)):
            gain = self.calMarginE(tmp[j], E)
            if gain > best_gain:
                best_gain=gain
                picked = tmp[j]
        print('one more for class '+str(c)+' is picked, gain is '+str(best_gain))
        return picked

    def getCandidate(self):
        # class0_idx = list(self.true_label[0])
        # class1_idx = list(self.true_label[1])
        # candidate = []
        # for i in range(self.n_subgraph):
        #     tmp = [np.sum(self.distribution[:,i][class0_idx]), np.sum(self.distribution[:,i][class1_idx])]
        #     candidate.append((i,tmp.index(max(tmp))))
        candidate = [(x, self.score[x].index(max(self.score[x]))) for x in range(len(self.score))]
        # candidate = [(x, self.score[0][x]) for x in range(len(self.score[0]))]

        return candidate

    def getDis(self):
        class0_idx = list(self.true_label[0])
        class1_idx = list(self.true_label[1])
        dis = []
        for i in range(self.n_subgraph):
            dis.append([np.sum(self.distribution[:,i][class0_idx]), np.sum(self.distribution[:,i][class1_idx])])
        return dis


    def explain(self, k, target_class = 1, oneMore=False, par_test = False, test_base = None):
        if self.dataset=='MUTAG' and target_class==0:
            print('For MUTAG dataset, only explain mutagenic class.')
            return

        if par_test:
            candidate = test_base
        else:
            candidate = [x for x in self.candidate if x[1]==target_class]

        E = []
        objective = 0
        i = 0
        while i<k:
        # for i in range(k):
            print('-------------------- iter: '+str(i))
            M = []
            for j in range(len(candidate)):
                print('-------------------- iter: ' + str(i) + 'element # ' +str(j))
                gain = self.calMarginE(candidate[j], E)
                if gain>=0:
                    if len(M)>=k :
                        if gain>M[-1][-1]:
                            M.pop() # pop last element, the smallest one
                            M.append([candidate[j],gain])
                    else:
                        M.append([candidate[j], gain])
                        print('Not yet k, added')
                M.sort(key = lambda x:x[-1], reverse= True) # decreasing
            if len(M) ==0:
                print('No positive gain! Greedy terminated.', end='\n\n')
                # if len(E) ==0:
                #     print('+++ no explanantion is generated, rerun.')
                #     return self.explain(k=k, target_class = target_class)

                return E
            print('-------------------- iter: ' + str(i) + ' len of M is ' + str(len(M)))
            print('-------------------- iter: ' + str(i) + ' min marginal is ' + str(M[-1][-1]))
            M+=[[0]] * (k-len(M))
            picked = random.choice(M)
            while picked[-1]==0 and i<k:
                print('-------------------- iter: '+str(i)+' picked a dummy')
                picked = random.choice(M)
                i+=1
            # while picked >=len(M) and i<k:
            #     print('-------------------- iter: '+str(i)+' picked ('+str(picked)+', is a dummy')
            #     picked = random.choice(list(range(k)))
            #     i+=1
            if i==k:                        
            #     if len(E) ==0:
            #         print('+++ no explanantion is generated, rerun.')
            #         return self.explain(k=k, target_class = target_class)
                return E
            print('-------------------- iter: '+str(i) + ' picked ('+str(picked[0][0])+', '+str(picked[0][1])+')')
            # print('margin to be added: ')
            # print(self.calMarginE(picked[0], E, True))
            objective += picked[-1]
            # print('Current objective: '+str(objective))
            E.append(picked[0])
            i+=1

        # if oneMore:
        #     for c in range(self.n_class):
        #         if len([e for e in E if e[-1]==c])==0:
        #             print('Class '+str(c) +' exp is missing')
        #             E.append(self.oneMore(E, candidate, c))

        # if len(E) ==0:
        #     print('+++ no explanantion is generated, rerun.')
        #     return self.explain(k=k, target_class = target_class)
        
        # if self.save_path is None:
        #     print('no save_path')
        #     if not os.path.isdir('result'):
        #         os.mkdir('result')
        #     if not os.path.isdir(os.path.join('result', self.dataset)):
        #         os.mkdir(os.path.join('result', self.dataset))

        #     self.save_path = 'result/'+self.dataset+'/'+self.model._get_name()+'_result/'
        #     if not os.path.isdir(self.save_path):
        #         os.mkdir(self.save_path)

        #     if self.dataset=='isAcyclic':
        #         self.save_path = 'result/'+self.dataset+'/'+self.model._get_name()+'_result/'+str(self.isAcyclic_n_nodes)+'_nodes/'
        #         if not os.path.isdir(self.save_path):
        #             os.mkdir(self.save_path)            
        return E

    def repeatExplain(self, k, repeat, target_class, par_test = False, test_base = None, save = False):
        # if not os.path.isdir('result'):
        #     os.mkdir('result')
        # if not os.path.isdir(os.path.join('result', self.dataset)):
        #     os.mkdir(os.path.join('result', self.dataset))

        # self.save_path = 'result/'+self.dataset+'/'+self.model._get_name()+'_result/'
        # if not os.path.isdir(self.save_path):
        #     os.mkdir(self.save_path)

        # if self.dataset=='isAcyclic':
        #     self.save_path = 'result/'+self.dataset+'/'+self.model._get_name()+'_result/'+str(self.isAcyclic_n_nodes)+'_nodes/'
        #     if not os.path.isdir(self.save_path):
        #         os.mkdir(self.save_path)

        exp_output = []
        # exp_population = set()
        # while len(exp_output)<repeat:
        for r in range(repeat):
            # print('======= working on repeat: '+str(r))
            if par_test:
                exp_output.append(self.explain(k, target_class=target_class, par_test = True, test_base = test_base))
                # exp_output.append(self.explain_silence(k, par_test = True, test_base = test_base))
            else:
                # exp_output.append(self.explain_silence(k))
                # E = self.explain(k, target_class=target_class)
                # if len(E)!=0:
                    # exp_output.append(E)
                exp_output.append(self.explain(k, target_class=target_class))
            # exp_population = exp_population.union(set(exp_output[-1]))
        # print('ALL explanation sets generated :')
        # print(exp_output)

        if save:
            # self.exp_population = exp_population

            output = self.generateOutput(exp_output)
            # print(self.evalOutput(output))
            self.evalMultipleRuns(exp_output)

            exp_output.append(['parameters']+self.Lambda.tolist())
            # with open(self.gSpan_output_file.replace('gSpan_output','class'+str(target_class)+'_all_exp.json'), 'w') as f:
            with open(self.save_path+'class_'+str(target_class)+'_all_exp.json', 'w') as f:
                json.dump(exp_output, f)
            # exp_output.pop(-1)
            # output = self.generateOutput(exp_output, exp_population)
            # with open(self.save_path+'class_'+str(target_class)+'_evalResult.json', 'w') as f:
            #     json.dump(self.evalOutput(output), f)

        return exp_output, output

    def generateOutput(self, exp_output, fractional = True, diversity_weight = False):
        frac_vote = {}
        for exp in exp_output:
            for e in exp:
                if e not in frac_vote:
                    frac_vote[e]=0
                frac_vote[e]+=1/len(exp)

        count_vote = []
        for exp in exp_output:
            if len(exp)==0:
                continue
            if diversity_weight:
                if fractional:
                    count_vote.append(sum([self.diver_dic[x[0]]*frac_vote[x] for x in exp])/len(exp))
                else:
                    count_vote.append(sum([self.diver_dic[x[0]]*frac_vote[x] for x in exp]))
            else:
                if fractional:
                    count_vote.append(sum([frac_vote[x] for x in exp])/len(exp))
                else:
                    count_vote.append(sum([frac_vote[x] for x in exp]))

        # print('the max vote is '+str(max(count_vote)))
        if len(count_vote)!=0:
            return exp_output[count_vote.index(max(count_vote))]
        else:
            return []

    def evalIndividualExp(self, e):

        support = np.count_nonzero(self.distribution[:,e[0]][list(self.true_label[e[-1]])])
        denial = np.count_nonzero(self.distribution[:,e[0]][list(self.true_label[1-e[-1]])])

        edges = self.size_dict[e[0]]['e']

        return self.score[e[0]][e[1]], support/(support+denial), denial/(support+denial), edges

    def evalMultipleRuns(self, all_exp_input):

        all_exp = [x for x in all_exp_input if len(x)!=0]

        # print('all_exp')
        # print(all_exp)

        ind_run_eval = []
        for exp in all_exp:
            ind_run_eval.append(self.evalOutput(exp))

        multi_run_result = np.mean(np.asarray(ind_run_eval), axis=0).tolist()

        # multi_run_result = []
        # for exp in all_exp:
        #   # nodes = 0
        #   edges = 0
        #   for e in exp:
        #       # nodes+=self.size_dict[str(e[0])]['n']
        #       edges+=self.size_dict[e[0]]['e']
        #   # multi_run_result.append([len(exp)]+ self.evalOutput(exp)[1:4]+[nodes/len(exp), edges/len(exp)])
        #   multi_run_result.append([len(exp)]+ self.evalOutput(exp)[1:4]+[edges/len(exp)])

        # final_result = []
        # for i in range(len(multi_run_result[0])):
        #   final_result.append(sum([x[i] for x in multi_run_result])/len(multi_run_result))

        target_class= all_exp[0][0][-1]
        with open(self.save_path+'/class_'+str(target_class)+'_mulirun_quantitatives.json','w') as f:
          json.dump(multi_run_result,f)
        # print('multile run result')
        # print(multi_run_result)

    def calObj(self, output): # for exhaustive search for optima

        F = sum([self.score[x[0]][x[1]] for x in output])/len(output)

        sup_distribution = self.distribution[list(self.true_label[output[0][1]]),:]
        S = np.count_nonzero(np.sum(sup_distribution[:, [x[0] for x in output]], axis=1))/self.n_total_inst

        den_distribution = self.distribution[list(self.true_label[1-output[0][1]]),:]
        avg_den = np.sum(den_distribution[:, [x[0] for x in output]])/self.n_total_inst
        D = 1-avg_den/self.n_subgraph

        Z = 1-len(output)/self.n_subgraph

        score = self.Lambda[0]*F + self.Lambda[1]*S + self.Lambda[2]*D + self.Lambda[-1]*Z
        return score/self.Lambda[0]
        
    # def evalAllClassOutput(self, output):
    #     # process output and evaluate
    #     print('############### evaluation')

    #     output_0 = list([x for x in output if x[-1]==0])
    #     output_1 = list([x for x in output if x[-1]==1])
    #     # if len(output_0)==0:
    #     #     print('Warning: 0 explanation for class 0')
    #     print('explanantion for class 0')
    #     print(output_0)

    #     print('explanantion for class 1')
    #     print(output_1)

    #     # evaluation
    #     # Size of global explanation set
    #     print('Total no. explanation generated: ')
    #     print(len(output_0+output_1))

    #     evaluation = self.evalStat(output_0+output_1)
    #     print('Recognition, coverage(0,1) , disagreement(0,1), overlap')
    #     print(evaluation)
    #     print('Objective: expressiveness, support, denial, in-class co-orr, cross-class co-orr, comprehensiveness, size. Lambda: ')
    #     print(self.Lambda)

    # def singleTuning(self, pos, test_base, k ,r):
    #     if self.Lambda[pos]==0:
    #         step = 10
    #     else:
    #         step = .5*self.Lambda[pos]
    #     exp_output, exp_population = self.repeatExplain(k, r, par_test = True, test_base = test_base)
    #     output_withWeight = self.generateOutput(exp_output, exp_population, diversity_weight = True, diver_dic = self.diver_dic)
    #     eval_metric = self.evalStat(output_withWeight)
    #     denial = (self.n_inst_clss[0]*eval_metric[3]+self.n_inst_clss[1]*eval_metric[4])/self.n_total_inst
    #     sign = 1
    #     c = 0
    #     while denial<0.02 and eval_metric[-1]<0.005 and len(output_withWeight)<=.1*self.n_total_inst:
    #         old_eval = eval_metric
    #         self.Lambda[pos]+= sign*step
    #         exp_output, exp_population = self.repeatExplain(k, r, par_test=True, test_base=test_base)
    #         output_withWeight = self.generateOutput(exp_output, exp_population, diversity_weight=True,
    #                                                 diver_dic=self.diver_dic)
    #         eval_metric = self.evalStat(output_withWeight)
    #         denial = (self.n_inst_clss[0]*eval_metric[3]+self.n_inst_clss[1]*eval_metric[4])/self.n_total_inst
    #         c+=1
    #         if eval_metric[0]<old_eval[0]:
    #             sign = sign*(-1)
    #     print('Finished tuning pos '+str(pos)+', took '+str(c)+' steps.')
    #     return c

    # def parTuning(self, k, r):
    #     test_base = random.sample([(x,0) for x in range(self.n_subgraph)]+[(x,1) for x in range(self.n_subgraph)], int(0.05*2*self.n_subgraph))
    #     self.test_base = test_base
    #     times = 100
    #     while times>3:
    #         times = 0
    #         fold = list(range(len(self.Lambda)))
    #         fold.remove(fold[-2])
    #         while len(fold)>0:
    #             pos = random.choice(fold)
    #             times+=self.singleTuning(pos, test_base,k, r)
    #             fold.remove(pos)
    #     print('Finished tuning! Tuned Lambda is')
    #     print(self.Lambda)


if __name__ == "__main__":
    path = 'data/MUTAG/'
    gSpan_output = 'MUTAG_data_no_edge_s9_l4_u9_gSpan'
    score_file = 'MUTAG_data_no_edge_s9_l4_u9_gSpan.json'
    model = DAG(path=path, gSpan_output=gSpan_output, score_file=score_file, n_class=2, n_inst_clss=[63, 125])
    model.Lambda = np.array([10000, 10000, 10000, model.n_total_inst*model.n_subgraph*model.n_subgraph/100000, model.n_total_inst*model.n_subgraph*model.n_subgraph/10000000, 10, model.size_D/10], dtype=np.int64)
    # k = 2*model.n_subgraph
    # # k = 100
    # model.evalOutput(model.explain(k))
    # print('k = '+str(k))
