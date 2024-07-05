import os
import csv
import pandas as pd
import torch
import copy
import random
from collections import defaultdict
from structure.dataloader import RuleDataset, Iterator
from torch.utils.data import Dataset, DataLoader
import pyMakex
from structure.salepredictor import SalePredictor
import structure.Convert as Convert
import structure.Match as Match
from structure.structure import Pattern
from structure.structure import REP
from collections import defaultdict


class Makex:
    def __init__(self, args, graph, x_node_label, y_node_label, q_label):
        self.args = args
        self.graph = graph
        self.num_relations = self.graph.NumOfEncode()
        self.encode_map = self.graph.GetEncodeMap()
        self.inv_encode_map = {}
        self.weight2_ = None
        for i in range(len(self.encode_map)):
            edge_label, dst_label = self.encode_map[i]
            self.inv_encode_map[(dst_label, edge_label)] = i
        self.x_node_label = x_node_label
        self.y_node_label = y_node_label
        self.q_label = q_label
        self.x_adj_code_list = self.graph.AdjEncodeList(self.x_node_label)
        self.y_adj_code_list = self.graph.AdjEncodeList(self.y_node_label)
        self.sale_predictor = SalePredictor(self.args.top_k, self.args.sort_criteria, self.args.num_process)
        self.pattern_set = set()



def read_csv_check_header(file_path, columns):
    first_row = pd.read_csv(file_path, nrows=1)
    
    if 'graph_id' in first_row.columns:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path, header=None, names=columns)
        
    return df


def read_subgraph(subgraph_path, subgraph_pivot_path, edge_label_reverse_csv):

    edge_reverse_csv = pd.read_csv(edge_label_reverse_csv)

    src_reverse_dict = dict(zip(edge_reverse_csv['src_type'], edge_reverse_csv['reverse_type']))

    subgraph_pivot_columns = ['graph_id', 'pivot_x', 'pivot_y']
    subgraph_columns = ['graph_id', 'src_id', 'dst_id', 'src_type', 'dst_type', 'edge_type']
    subgraph_data = read_csv_check_header(subgraph_path, subgraph_columns)

    subgraph_data_pivot = read_csv_check_header(subgraph_pivot_path, subgraph_pivot_columns)
    subgraph_data_all = subgraph_data.merge(subgraph_data_pivot, on='graph_id', how='left')

    explain_graph_data = {}
    for _, row in subgraph_data_all.iterrows():
        graph_id = row['graph_id']
        if graph_id not in explain_graph_data:
            explain_graph_data[graph_id] = {
                'pivot': {},
                'neighbors_dict': {},
                'node_type': {},
                'edge_type_dict': {},
            }
    
        src_id, dst_id = int(row['src_id']), int(row['dst_id'])
        pivot = explain_graph_data[graph_id]['pivot']
        neighbors_dict = explain_graph_data[graph_id]['neighbors_dict']
        node_type = explain_graph_data[graph_id]['node_type']
        edge_type_dict = explain_graph_data[graph_id]['edge_type_dict']

        pivot['pivot_x'] = int(row['pivot_x'])
        pivot['pivot_y'] = int(row['pivot_y'])
        if src_id not in neighbors_dict:
            neighbors_dict[src_id] = []
        if dst_id not in neighbors_dict:
            neighbors_dict[dst_id] = []

        if src_id == int(row['pivot_x']) and row['src_type'] != 1:
            continue
        if src_id == int(row['pivot_y']) and row['src_type'] != 0:
            continue
        if dst_id == int(row['pivot_x']) and row['dst_type'] != 1:
            continue
        if dst_id == int(row['pivot_y']) and row['dst_type'] != 0:
            continue
        if not ((src_id == int(row['pivot_x']) and dst_id == int(row['pivot_y'])) or
            (src_id == int(row['pivot_y']) and dst_id == int(row['pivot_x']))):    
            if dst_id not in neighbors_dict[src_id]:
                neighbors_dict[src_id].append(dst_id)
            if src_id not in neighbors_dict[dst_id]:
                neighbors_dict[dst_id].append(src_id)

        if src_id not in node_type.keys():
            if src_id == int(row['pivot_x']):
                node_type[src_id] = 1
            elif src_id == int(row['pivot_y']):
                node_type[src_id] = 0
            else:
                node_type[src_id] = row['src_type']
        if dst_id not in node_type.keys():
            if dst_id == int(row['pivot_x']):
                node_type[dst_id] = 1
            elif dst_id == int(row['pivot_y']):
                node_type[dst_id] = 0
            else:
                node_type[dst_id] = row['dst_type']
        
        edge_key = (src_id, dst_id)
        edge_type_dict[edge_key] = row['edge_type']
        reverse_edge_key = (dst_id, src_id)
        edge_type_dict[reverse_edge_key] = src_reverse_dict[row['edge_type']]

    return explain_graph_data




def random_walk_degree_length(neighbors_dict_ori, pivot_x, pivot_y, max_degree, max_length):
    neighbors_dict = {key: value[:] for key, value in neighbors_dict_ori.items()}
    star_x = {}
    star_y = {}

    for degree_iter in range(max_degree): 
        
        start_node_x = int(pivot_x)
        start_node_y = int(pivot_y)

        star_x[degree_iter] = [start_node_x]
        star_y[degree_iter] = [start_node_y]

        x_leaf_node = False
        y_leaf_node = False
        pattern_x_leaf_node = []
        pattern_y_leaf_node = []
        node_x_status = True
        node_y_status = True
        
        iter_ = 0
        iteration = 2
        iter_max = 10
        iteration_max = 100
        while(len(star_x[degree_iter]) != max_length+1 or len(star_y[degree_iter]) != max_length+1):
            
            start_node_x = int(pivot_x)
            start_node_y = int(pivot_y)

            

            star_x[degree_iter] = [start_node_x]
            star_y[degree_iter] = [start_node_y]

            x_leaf_node = False
            y_leaf_node = False
            pattern_x_leaf_node = []
            pattern_y_leaf_node = []
            node_x_status = True
            node_y_status = True

            iter_ += 1

            for length_iter in range(max_length):
                if node_x_status and not x_leaf_node:
                    if start_node_x in neighbors_dict and neighbors_dict[start_node_x]:
                        node_x_candidates = [node for node in neighbors_dict[start_node_x] if node not in [item for sublist in star_x.values() for item in sublist]]
                        if node_x_candidates:
                            node_x = random.choice(node_x_candidates)
                        else:
                            node_x_status = False

                        while(node_x_candidates):
                            if node_x in [item for sublist in star_y.values() for item in sublist]:
                                if node_x in pattern_y_leaf_node:
                                    x_leaf_node = True
                                    pattern_x_leaf_node.append(node_x)
                                    break
                                else:
                                    node_x_candidates.remove(node_x)
                                    if node_x_candidates:
                                        node_x = random.choice(node_x_candidates)
                                        continue
                                    else:
                                        x_leaf_node = True
                                        node_x_status = False
                                        pattern_x_leaf_node.append(star_x[degree_iter][-1])
                                        break
                            else:
                                break
                        
                        if node_x_status: 
                            star_x[degree_iter].append(node_x)
                            if start_node_x in neighbors_dict[node_x]:
                                neighbors_dict[node_x].remove(start_node_x)
                                neighbors_dict[start_node_x].remove(node_x)
                            if length_iter == max_length - 1:
                                x_leaf_node = True
                                pattern_x_leaf_node.append(node_x)
                            start_node_x = node_x

                if node_y_status and not y_leaf_node:
                    if start_node_y in neighbors_dict and neighbors_dict[start_node_y]:
                        node_y_candidates = [node for node in neighbors_dict[start_node_y] if node not in [item for sublist in star_y.values() for item in sublist]]
                        if node_y_candidates:
                            node_y = random.choice(node_y_candidates)
                        else:
                            node_y_status = False

                        while(node_y_candidates):
                            if node_y in [item for sublist in star_x.values() for item in sublist]:
                                if node_y in pattern_x_leaf_node:
                                    y_leaf_node = True
                                    pattern_y_leaf_node.append(node_y)
                                    break
                                else:
                                    node_y_candidates.remove(node_y)
                                    if node_y_candidates:
                                        node_y = random.choice(node_y_candidates)
                                        continue
                                    else:
                                        node_y_status = False
                                        y_leaf_node = True
                                        pattern_y_leaf_node.append(star_y[degree_iter][-1])
                                        break
                            else:
                                break
                        
                        if node_y_status:
                            star_y[degree_iter].append(node_y)
                            if start_node_y in neighbors_dict[node_y]:
                                neighbors_dict[node_y].remove(start_node_y)
                                neighbors_dict[start_node_y].remove(node_y)
                            if length_iter == max_length - 1:
                                y_leaf_node = True
                                pattern_y_leaf_node.append(node_y)
                            start_node_y = node_y

            if iter_ > iter_max:
                if len(star_x[degree_iter]) == 1 and iteration < iteration_max:
                    iter_ = 0
                    iteration = iteration * iteration
                else:
                    break

    return star_x, star_y



def sort_nested_list(nested_list):
    adj_list = defaultdict(list)
    for a, b, c in nested_list:
        adj_list[a].append((b, c))

    for key in adj_list:
        adj_list[key].sort()
    
    result = []
    visited = set()

    def dfs(node):
        while adj_list[node]:
            neighbor, value = adj_list[node].pop(0)
            if (node, neighbor, value) not in visited:
                visited.add((node, neighbor, value))
                result.append([node, neighbor, value])
                dfs(neighbor)

    for key in sorted(adj_list.keys()):
        if key not in visited:
            dfs(key)
    return result


def GetPatternFromFile(pattern_file):
    pattern_lines = []
    with open(pattern_file, 'r', encoding='ISO-8859-1') as pattern_file_reader:
        pattern_lines = pattern_file_reader.readlines()
    all_pattern = []
    for i in range(0, len(pattern_lines)):
        rule_line = pattern_lines[i]
        start_index_first = rule_line.index("[[")
        end_index_first = rule_line.index("]]") + 2
        start_index_second = rule_line.index("[[", end_index_first)
        end_index_second = rule_line.index("]]", start_index_second) + 2

        first_list = eval(rule_line[start_index_first:end_index_first])
        second_list = eval(rule_line[start_index_second:end_index_second])

        vertex_list = first_list
        edge_list = second_list

        pattern = Pattern(vertex_list, edge_list)
        all_pattern.append(pattern)
    return all_pattern


def SubgraphToPattern(star_x, star_y, node_type, edge_type, pivot_x, pivot_y):
    vertex_list = []
    edge_list = []
    vertex_new_mapping = {}

    node_id = 1

    for pivot in [pivot_x, pivot_y]:
        pivot_id = int(pivot)
        vertex_new_mapping[pivot_id] = node_id
        if pivot_id not in node_type.keys():
            continue
        else:
            node_label = node_type[pivot_id]
            vertex_list.append([node_id, int(node_label)])
            node_id += 1

    for star in [star_x, star_y]:
        for values in star.values():
            for value in values:
                if value not in vertex_new_mapping:
                    vertex_new_mapping[value] = node_id
                    node_label = node_type[value]
                    vertex_list.append([node_id, int(node_label)])
                    node_id += 1

    for star in [star_x, star_y]:
        for values in star.values():
            for i in range(len(values) - 1):
                src_id = int(values[i])
                dst_id = int(values[i + 1])
                key = (src_id, dst_id)
                if key not in edge_type:
                    print("this edge type not in edge_type.key: ", key)
                edge_label = edge_type.get(key, 0)
                edge_list.append([vertex_new_mapping[src_id], vertex_new_mapping[dst_id], int(edge_label)])

    if edge_list:
        return Pattern(vertex_list, edge_list)
    else:
        return None



def Score_Q(star, hop_decay_factor):
    score = 0
    for key in star:
        path_length = len(star[key])
        score *=  pow(hop_decay_factor, path_length)
    return score



def REP_make_hashable(lst):
    if isinstance(lst, list):
        return tuple(REP_make_hashable(e) for e in lst)
    else:
        return lst

def REP_make_list(tup):
    if isinstance(tup, tuple):
        return [REP_make_list(e) for e in tup]
    else:
        return tup



def RankingRuleForScore_Q_X(tmp_gen_rep_dict_list, Score_Q_dict):
    for dict_REP in tmp_gen_rep_dict_list:
        if dict_REP:
            for key, value in dict_REP.items():
                rep_list = REP_make_list(key)
                pattern_vertex = rep_list[0]
                pattern_edge = rep_list[1]
                pattern_key = REP_make_hashable([pattern_vertex, pattern_edge])
                score_q = Score_Q_dict[pattern_key]
                score_x = value['leaf_gini']
                value['pattern_score'] = score_q
                value['rule_score'] = score_q * score_x

    tmp_gen_rep_dict_list.sort(key=lambda x: x.get('rule_score', 0) if x else 0, reverse=True)

    return tmp_gen_rep_dict_list




def filter_duplicate_constraints(predicate_list):
    seen = set()
    cleaned_predicate_list = []
    for item in reversed(predicate_list):
        if item[0] == "Constant" and item[4] == "double":
            key = (item[1], item[2], item[5])
            if key not in seen:
                seen.add(key)
                cleaned_predicate_list.append(item)
        else:
            cleaned_predicate_list.append(item)
    
    cleaned_predicate_list = list(reversed(cleaned_predicate_list))
    return cleaned_predicate_list


def calculate_rep_gini_score(pattern_vertex, predicate_list, feature_importance):
    score_dict = {vertex[0]: 0 for vertex in pattern_vertex}

    for item in predicate_list:
        attribute_key = f""
        if item[0] == "Constant":
            vertex_id = item[1]
            attribute_name = item[2]
            attribute_value = str(item[3])
            attribute_type = item[4]
            if attribute_type == 'string' or attribute_type == 'int':
                attribute_key = f"{attribute_name}_{attribute_value}_id{vertex_id}"
            else:
                attribute_key = f"{attribute_name}_id{vertex_id}"
            
            if attribute_key in feature_importance:
                score_dict[vertex_id] += feature_importance[attribute_key]
        
        elif item[0] == "Variable":
            vertex_id1 = item[1]
            vertex_id2 = item[3]
            attribute_name = item[2]
            operation = item[5]
            if operation == '=' or operation == '!=':
                attribute_key = f"{attribute_name}_comp_0_id{vertex_id1}_id{vertex_id2}"
            if operation == '<' or operation == '>=':
                attribute_key = f"{attribute_name}_comp_1_id{vertex_id1}_id{vertex_id2}"
            if operation == '<=' or operation == '>':
                attribute_key = f"{attribute_name}_comp_2_id{vertex_id1}_id{vertex_id2}"

            if attribute_key in feature_importance:
                score = feature_importance[attribute_key]
                score_dict[vertex_id1] += score
                score_dict[vertex_id2] += score
    
    return score_dict

def get_edge_num_length(edge_list):
    dict_edge_length = {}
    node1_or_node2 = True
    node1_degree = 1
    node2_degree = 1

    for edge_index in range(len(edge_list)):
        if edge_index == 0:
            if edge_list[edge_index][0] == 1:
                dict_edge_length[(1, node1_degree)] = 1
                dict_edge_length[(2, node2_degree)] = 0
                node1_or_node2 = True
            else:
                dict_edge_length[(1, node1_degree)] = 0
                dict_edge_length[(2, node2_degree)] = 1
                node1_or_node2 = False
            visited = edge_list[edge_index][1]
            continue

        if edge_list[edge_index][0] == 2:
            if node1_or_node2:
                node1_or_node2 = False
                visited = 2

        if node1_or_node2:
            if edge_list[edge_index][0] == visited:
                dict_edge_length[(1,node1_degree)] += 1
            else:
                node1_degree += 1
                dict_edge_length[(1,node1_degree)] = 1

        if not node1_or_node2:
            if edge_list[edge_index][0] == visited:
                dict_edge_length[(2,node2_degree)] += 1
            else:
                node2_degree += 1
                dict_edge_length[(2,node2_degree)] = 1
                
        visited = edge_list[edge_index][1]
    return dict_edge_length
    



def sample_test_user_item_pair(file_name, pair_number, score_threshold):
    column_names = ["user_id", "item_id", "score"]
    df = pd.read_csv(file_name, header=None, names=column_names)
    filtered_df = df[df['score'] > score_threshold]
    sample_size = min(pair_number, len(filtered_df))
    selected_rows = filtered_df.sample(sample_size)
    result_list = list(zip(selected_rows['user_id'], selected_rows['item_id']))

    return result_list



def sample_test_pairs(file_name):
    df = pd.read_csv(file_name)
    sample_pairs = list(zip(df['user_id'], df['item_id']))
    return sample_pairs


def explanation_rep_to_csv(gnn_exp_model_input_explanation,pair_id,df_original_v, makex_explanation_v, makex_explanation_e):

    directory_path = os.path.dirname(makex_explanation_v)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    directory_path_e = os.path.dirname(makex_explanation_e)
    if not os.path.exists(directory_path_e):
        os.makedirs(directory_path_e)

    attributes_list = ['title', 'genres', 'avgrating', 'year', 'gender', 'age', 'occupation', 'zip-code', 'wllabel']
    attributes_type_list = ['title:string', 'genres:string', 'avgrating:double', 'year:double', 'gender:string', 'age:double', 'occupation:int', 'zip-code:string', 'wllabel:string']
    
    if not os.path.exists(makex_explanation_v):
        with open(makex_explanation_v, 'w', newline='') as f:
            writer = csv.writer(f)
            columns_vertex = ['pair_id', 'pivot_x', 'pivot_y', 'topk', 'rep_id', 'explanation_score', 'vertex_id', 'label_id:int'] + attributes_type_list
            writer.writerow(columns_vertex)

    if not os.path.exists(makex_explanation_e):
        with open(makex_explanation_e, 'w', newline='') as f:
            writer = csv.writer(f)
            columns_edge = ['pair_id', 'pivot_x', 'pivot_y', 'topk', 'rep_id', 'explanation_score', 'source_id:int', 'target_id:int', 'label_id:int']
            writer.writerow(columns_edge)

    topk = 0
    previous_rep_id = None
    for [explanation, rep_pattern_predicates, rep_id] in gnn_exp_model_input_explanation:
        if rep_id != previous_rep_id:
            topk = 0
            previous_rep_id = rep_id

        if 1 not in explanation[1] or 2 not in explanation[1]:
            continue

        edges_info = rep_pattern_predicates[1]

        for edge in edges_info:
            row_edge = []
            row_edge.append(int(pair_id))
            row_edge.append(int(explanation[1][1]))
            row_edge.append(int(explanation[1][2]))
            row_edge.append(int(topk))
            row_edge.append(int(rep_id))
            row_edge.append(float(explanation[0]))
            start_node = explanation[1][edge[0]]
            end_node = explanation[1][edge[1]]
            label = edge[2]
            row_edge.append(start_node)
            row_edge.append(end_node)
            row_edge.append(label)

            with open(makex_explanation_e, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_edge)


        predicates_list = rep_pattern_predicates[2]
        vertex_attributes = {}
        for predicates in predicates_list:
            if predicates[0] == 'Constant':
                pattern_id = predicates[1]
                key = explanation[1][pattern_id]
                if key not in vertex_attributes.keys():
                    vertex_attributes[key] = set()
                vertex_attributes[key].add(predicates[2])
            if predicates[0] == 'Variable':
                pattern_id_1 = predicates[1]
                pattern_id_2 = predicates[3]
                if predicates[2] != 'id':
                    key_id_1 = explanation[1][pattern_id_1]
                    key_id_2 = explanation[1][pattern_id_2]
                    if key_id_1 not in vertex_attributes.keys():
                        vertex_attributes[key_id_1] = set()
                    vertex_attributes[key_id_1].add(predicates[2])
                    if key_id_2 not in vertex_attributes.keys():
                        vertex_attributes[key_id_2] = set()
                    vertex_attributes[key_id_2].add(predicates[2])

        
        for index, value in enumerate(explanation):
            if isinstance(value, dict):
                for key, val in value.items():
                    row_vertex = []
                    row_vertex = [int(pair_id), int(explanation[1][1]), int(explanation[1][2]), int(topk), int(rep_id), float(explanation[0]), val]

                    vertex_info = df_original_v[df_original_v['vertex_id:int'] == val]
                    row_vertex.append(vertex_info['label_id:int'].iloc[0])
                    

                    for attribute in attributes_list:
                        if val in vertex_attributes and attribute in vertex_attributes[val]:
                            index = attributes_list.index(attribute)
                            attributes_type = attributes_type_list[index]
                            row_vertex.append(vertex_info[attributes_type].iloc[0])
                        else:
                            row_vertex.append('')

                    with open(makex_explanation_v, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row_vertex)
        topk += 1
    




def filter_node_same_predicates(predicate_list, predicate_score_list):
    new_predicate_list = []
    new_predicate_score_list = []
    dict_predicate_list = {}
    for i in range(len(predicate_list)):
        node_id = predicate_list[i][1]
        node_attributes = predicate_list[i][2]
        if node_attributes == 'wllabel':

            key = (predicate_list[i][3],node_attributes)
            if '**' in str(predicate_list[i][3]):
                if key not in dict_predicate_list.keys():
                    dict_predicate_list[key] = predicate_list[i]
                    variable_predicate_list = []
                    variable_ids = predicate_list[i][3].split('**')

                    node_id1 = int(variable_ids[0])
                    node_id2 = int(variable_ids[1])
                    variable_predicate_list.append('Variable')
                    variable_predicate_list.append(node_id1)
                    variable_predicate_list.append(node_attributes)
                    variable_predicate_list.append(node_id2)
                    variable_predicate_list.append(node_attributes)
                    variable_predicate_list.append(predicate_list[i][5])


                    new_predicate_list.append(variable_predicate_list)
                    new_predicate_score_list.append(predicate_score_list[i])
            else:
                key = (node_id, node_attributes)
                if key not in dict_predicate_list.keys():
                    dict_predicate_list[key] = predicate_list[i]
                    new_predicate_list.append(predicate_list[i])
                    new_predicate_score_list.append(predicate_score_list[i])

        else:
            key = (node_id, node_attributes)
            if key not in dict_predicate_list.keys():
                dict_predicate_list[key] = predicate_list[i]
                new_predicate_list.append(predicate_list[i])
                new_predicate_score_list.append(predicate_score_list[i])
    return new_predicate_list, new_predicate_score_list


def filter_node_same_predicates_wllabel(predicate_list, predicate_score_list):
    new_predicate_list = []
    new_predicate_score_list = []
    dict_predicate_list = {}
    for i in range(len(predicate_list)):
        node_id = predicate_list[i][1]
        node_attributes = predicate_list[i][2]
        if node_attributes == 'wllabel_select':

            variable_ids = predicate_list[i][3].split('**')

            node_id1 = int(variable_ids[0])
            node_id2 = int(variable_ids[1])
            wllabel_value = variable_ids[2]
            node_attributes_ = "wllabel"


            key_node1 = (node_id1,node_attributes_)
            
            if key_node1 not in dict_predicate_list.keys():
                dict_predicate_list[key_node1] = node_attributes_

                constant_predicate_list = []
                constant_predicate_list.append('Constant')
                constant_predicate_list.append(node_id1)
                constant_predicate_list.append(node_attributes_)
                constant_predicate_list.append(wllabel_value)
                constant_predicate_list.append('string')
                constant_predicate_list.append('=')

                new_predicate_list.append(constant_predicate_list)
                new_predicate_score_list.append(predicate_score_list[i])


            key_node2 = (node_id2,node_attributes_)    
            if key_node2 not in dict_predicate_list.keys():
                dict_predicate_list[key_node2] = node_attributes_
                constant_predicate_list2 = []

                constant_predicate_list2.append('Constant')
                constant_predicate_list2.append(node_id2)
                constant_predicate_list2.append(node_attributes_)
                constant_predicate_list2.append(wllabel_value)
                constant_predicate_list2.append('string')
                constant_predicate_list2.append('=')

                new_predicate_list.append(constant_predicate_list2)
                new_predicate_score_list.append(predicate_score_list[i])
        else:
            key = (node_id, node_attributes)
            if key not in dict_predicate_list.keys():
                dict_predicate_list[key] = predicate_list[i]
                new_predicate_list.append(predicate_list[i])
                new_predicate_score_list.append(predicate_score_list[i])
    return new_predicate_list, new_predicate_score_list



def get_all_REP(rep_file, each_pattern_rep):
    rule_lines = []
    with open(rep_file, 'r', encoding='ISO-8859-1') as rule_file:
        rule_lines = rule_file.readlines()

    all_rep = []
    viewed_rep = {}
    viewed_rep_all = {}
    for i in range(0, len(rule_lines)):
        rule_line = rule_lines[i]
        rule_line = rule_line.strip()
        start_idx = rule_line.find('[')
        rule_line = rule_line[start_idx:]
        rep_rule = eval(rule_line)

        vertex_list = rep_rule[0]
        edge_list = rep_rule[1]
        key = str(vertex_list + edge_list)
        
        if key not in viewed_rep.keys():
            viewed_rep[key] = 1
            viewed_rep_all[key] = 1
        else:
            viewed_rep_all[key] += 1
            times = viewed_rep[key]
            if times >= each_pattern_rep:
                continue
            else:
                viewed_rep[key] += 1

        old_predicate_list = rep_rule[2]
        old_predicate_score_list = rep_rule[3]
        predicate_list, predicate_score_list = filter_node_same_predicates_wllabel(old_predicate_list, old_predicate_score_list)
        x_id_y_id_like_score = rep_rule[4]

        pattern = Pattern(vertex_list, edge_list)
        rep = REP(x_id_y_id_like_score[3], pattern, predicate_list, predicate_score_list, x_id_y_id_like_score[0], x_id_y_id_like_score[1], x_id_y_id_like_score[2])
        all_rep.append(rep)

    all_rep_sorted = sorted(all_rep, key=lambda rep: rep.score, reverse=True)
    return all_rep_sorted

    



def get_all_REP_filter_support_conf(rep_file, each_pattern_rep, pattern_num, sopport, conf, reserved_rep_file_supp_conf):
    rule_lines = []
    with open(rep_file, 'r', encoding='ISO-8859-1') as rule_file:
        rule_lines = rule_file.readlines()

    all_rep = []
    viewed_rep = {}
    viewed_rep_all = {}
    selected_rep_supp_conf = {}
    for i in range(0, len(rule_lines)):
        if(len(viewed_rep.keys()) >= pattern_num):
            break

        rule_line = rule_lines[i]
        rule_line = rule_line.strip()
        start_idx = rule_line.find('[')
        rule_line = rule_line[start_idx:]
        rep_rule = eval(rule_line)

        vertex_list = rep_rule[0]
        edge_list_no_sort = rep_rule[1]
        edge_list = sort_nested_list(edge_list_no_sort)
        support_conf = rep_rule[5]
        key = str(vertex_list + edge_list)
        
        if key not in viewed_rep.keys():
            viewed_rep[key] = 1
            viewed_rep_all[key] = 1
        else:
            viewed_rep_all[key] += 1
            times = viewed_rep[key]
            if times >= each_pattern_rep:
                continue
            else:
                viewed_rep[key] += 1

        old_predicate_list = rep_rule[2]
        old_predicate_score_list = rep_rule[3]
        predicate_list, predicate_score_list = filter_node_same_predicates_wllabel(old_predicate_list, old_predicate_score_list)
        key_supp_conf = str(vertex_list + edge_list + predicate_list + predicate_score_list)
        selected_rep_supp_conf[key_supp_conf] = support_conf
        x_id_y_id_like_score = rep_rule[4]

    
        pattern = Pattern(vertex_list, edge_list)
        rep = REP(x_id_y_id_like_score[3], pattern, predicate_list, predicate_score_list, x_id_y_id_like_score[0], x_id_y_id_like_score[1], x_id_y_id_like_score[2])
        all_rep.append(rep)

    all_rep_filter_by_supp_conf = []
    reserve_rep_file_os = open(reserved_rep_file_supp_conf,'w',encoding='ISO-8859-1')

    index = 0
    for key_supp_conf in selected_rep_supp_conf.keys():
        support_conf = selected_rep_supp_conf[key_supp_conf]
        if support_conf[0] < sopport or support_conf[1] < conf:
            continue
        else:
            all_rep_filter_by_supp_conf.append(all_rep[index])
            reserve_rep_file_os.write(str(all_rep[index].GetREPMatchArg()) + " [" + str(support_conf[0]) + ", " + str(support_conf[1]) + "]\n")
            reserve_rep_file_os.flush()
        index += 1

    return all_rep_filter_by_supp_conf

