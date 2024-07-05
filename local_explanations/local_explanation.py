import argparse
import pandas as pd
import random
from structure.structure import DataGraph
import pyMakex
import time
import multiprocessing
from Makex import Makex
from Makex import read_subgraph, SubgraphToPattern, Score_Q, random_walk_degree_length, REP_make_hashable
from Makex import REP_make_hashable, sample_test_pairs, explanation_rep_to_csv, get_all_REP
from structure.structure import REP
from structure.structure import Pattern
import os
import csv

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Makex", usage="run.py [<args>] [-h | --help]")

    parser.add_argument("--cuda", action="store_true", help="Use GPU.")

    # data graph and model
    parser.add_argument(
        "--v_path",
        type=str,
        default="./v.csv",
        help="vertex file of train graph",
    )
    parser.add_argument(
        "--e_path",
        type=str,
        default="./e.csv",
        help="edge file of train graph",
    )

    parser.add_argument(
        "--subgraph_path",
        type=str,
        default="./subgraph.csv",
        help="generate subgraphs",
    )

    parser.add_argument(
        "--subgraph_pivot_path",
        type=str,
        default="./subgraph_pivot.csv",
        help="generate subgraphs id and pivot id",
    )

    parser.add_argument(
        "--has_ml",
        type=int,
        default=1,
        help="has ml train data,1 is has,0 is not",
    )
    parser.add_argument(
        "--ml_path",
        type=str,
        default="./train.log",
        help="result file of ml",
    )
    
    parser.add_argument(
        "--delta_l",
        type=float,
        default=1.0,
        help="delta l of rep",
    )
    parser.add_argument(
        "--delta_r",
        type=float,
        default=1.0,
        help="delta r of rep",
    )

    parser.add_argument(
        "--user_offset",
        type=int,
        default=0,
        help="user_offset",
    )

    # user,item,like
    parser.add_argument("--user_label", type=int, default=1, help="user label of graph")
    parser.add_argument("--item_label", type=int, default=0, help="item label of graph")
    parser.add_argument("--like_label", type=int, default=1, help="like label of graph")

    parser.add_argument(
        "--number_of_sample_pair",
        type=int,
        default=500,
        help="number of sample pair for positive pairs and negative pairs",
    )
    parser.add_argument("--number_of_match", type=int, default=10, help="number of match for each pair")

    parser.add_argument("--top_k", type=int, default=150, help="number of select high score rep")

    parser.add_argument("--num_process", default=1, type=int, help="Set the number of process for training.")
    parser.add_argument("--iteration", default=4, type=int, help="Set the number of training iterations.")
    parser.add_argument("--max_predicates_length", default=20, type=int, help="Set the depth of DecisionTree.")
    parser.add_argument("--REP_num_of_one_pattern",
                        default=2000,
                        type=int,
                        help="Set the number of REPs when the decision tree returns.")
    parser.add_argument("--leaf_node_datanum", default=10, type=int, help="Set the number of leaf node's data num.")


    parser.add_argument("--sort_criteria", type=str, default="conf", help="criteria for sort reps. value is conf or f1")
    parser.add_argument("--supp_limit", type=int, default=80, help="limit for supp of rep")
    parser.add_argument("--conf_limit", type=float, default=0.6, help="limit for conf of rep")

    parser.add_argument(
        "--rep_num",
        type=int,
        default=10000,
        help="rep num remain",
    )

    parser.add_argument(
        "--pattern_num",
        type=int,
        default=100,
        help="rep num remain",
    )
    
    parser.add_argument(
        "--max_degree",
        type=int,
        default=2,
        help="subgraph to star pattern max degree",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=2,
        help="subgraph to star pattern max length",
    )

    parser.add_argument(
        "--max_subgraph_to_pattern_num",
        type=int,
        default=5,
        help="max num of star pattern sampled by subgraph",
    )

    parser.add_argument(
        "--hop_decay_factor",
        type=float,
        default=0.8,
        help="pattern node's hop decay factor ",
    )


    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="topk explanation size",
    )

    parser.add_argument(
        "--enable_topk",
        type=int,
        default= 1,
        help="topk explanation programe",
    )

    parser.add_argument(
        "--explanation_user_id",
        type=int,
        default= 9745,
        help="explnantion user",
    )

    parser.add_argument(
        "--explanation_item_id",
        type=int,
        default= 141,
        help="explnantion item",
    )

    parser.add_argument(
        "--test_sample_pairs_file",
        type=str,
        default="./hgt_sample_pairs.csv",
        help="test pair for GNN model",
    )

    parser.add_argument(
        "--makex_explanation_v",
        type=str,
        default="./makex_explanation_v.csv",
        help="makex explanation result file",
    )

    parser.add_argument(
        "--makex_explanation_e",
        type=str,
        default="./makex_explanation_e.csv",
        help="makex explanation result file",
    )


    parser.add_argument(
        "--sample_pair_num",
        type=int,
        default= 1000,
        help="sample pair num",
    )

    parser.add_argument(
        "--pattern_file",
        type=str,
        default="pattern.txt",
        help="all random walk pattern file",
    )

    parser.add_argument(
        "--rep_file",
        type=str,
        default="Rep.txt",
        help="all rep file",
    )

    parser.add_argument(
        "--edge_label_reverse_csv",
        type=str,
        default="edge_label_reverse.csv",
        help="reverse edge label",
    )

    parser.add_argument(
        "--reserve_rep_file",
        type=str,
        default="Rep.txt",
        help="all rep file",
    )

    parser.add_argument(
        "--topk_rep_id_file",
        type=str,
        default="Rep.txt",
        help="each pair match topk rep id file",
    )


    parser.add_argument(
        "--each_pattern_rep",
        type=int,
        default="3",
        help="each rep duplicate times",
    )

    parser.add_argument(
        "--rep_num_ratio",
        type=float,
        default="1.0",
        help="use rep num for local",
    )


    return parser.parse_args(args)


def main(args):

    subgraph_path = args.subgraph_path
    subgraph_pivot_path = args.subgraph_pivot_path

    max_degree = args.max_degree
    max_length = args.max_length
    pattern_num = args.max_subgraph_to_pattern_num

    hop_decay_factor = args.hop_decay_factor

    explain_graph_data = read_subgraph(subgraph_path, subgraph_pivot_path)

    gen_pattern_list = []
    Score_Q_dict = {}
    for graph_id in explain_graph_data:
        pivot_x = explain_graph_data[graph_id]['pivot']['pivot_x']
        pivot_y = explain_graph_data[graph_id]['pivot']['pivot_y']
        neighbors_dict = explain_graph_data[graph_id]['neighbors_dict']
        node_type = explain_graph_data[graph_id]['node_type']
        edge_type_dict = explain_graph_data[graph_id]['edge_type_dict']
        edge_type_label_dict = explain_graph_data[graph_id]['edge_type_label_dict']



        for pattern_iter in range(pattern_num):
            star_x, star_y = random_walk_degree_length(neighbors_dict, pivot_x, pivot_y, max_degree, max_length)

            score_Q = Score_Q(star_x, hop_decay_factor) + Score_Q(star_y, hop_decay_factor)
            pattern = SubgraphToPattern(star_x, star_y, node_type, edge_type_dict, edge_type_label_dict, pivot_x, pivot_y)

            if pattern is not None:
                gen_pattern_list.append(pattern)
                key = REP_make_hashable([pattern.vertex_list, pattern.edge_list])
                Score_Q_dict[key] = score_Q



    graph_pointer = pyMakex.ReadDataGraph(args.v_path, args.e_path)
    graph = DataGraph(graph_pointer)
    
    if args.has_ml == 1:
        pyMakex.ReadML(graph_pointer, args.ml_path, args.delta_l, args.delta_r, args.user_offset)
        print("positive pair count = ", graph.PositivePairNum())
        print("negative pair count = ", graph.NegativePairNum())


    makex = Makex(args, graph, args.user_label, args.item_label, args.like_label)

    

    for round in range(args.iteration):

        gen_rep_list = []
        all_rep_file = args.rep_file

        gen_rep_list = get_all_REP(all_rep_file, args.each_pattern_rep)


        rep_remain_num = args.rep_num
        gen_rep_list = gen_rep_list[0:min(rep_remain_num,len(gen_rep_list))]

        test_sample_pairs_file = args.test_sample_pairs_file
        sample_pairs_all = sample_test_pairs(test_sample_pairs_file)

        topk_rep_id_file = args.topk_rep_id_file
        topk = args.topk
        enable_topk = args.enable_topk
        sample_pair_num = args.sample_pair_num
        sample_pairs = sample_pairs_all[0:min(sample_pair_num, len(sample_pairs_all))]


        topk_list = [topk]
        for topk in topk_list:
            pair_id = 0
            df_original_v = pd.read_csv(args.v_path)

            all_rep_all_match_pair_num = 0

            directory_topk_rep_id_file = os.path.dirname(topk_rep_id_file)
            if not os.path.exists(directory_topk_rep_id_file):
                os.makedirs(directory_topk_rep_id_file)


            if not os.path.exists(topk_rep_id_file):
                with open(topk_rep_id_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    columns_vertex = ['pair_id', 'pivot_x', 'pivot_y', 'topk', 'rep_id']
                    writer.writerow(columns_vertex)



            for (user_id ,item_id) in sample_pairs:
                topk_explanation = []
                topk_explanation_scores_list = []
                topk_explanation_rep_id = []
                topk_explanation_rep_id_has_match = []

                sorted_topk_explanation_rep_id = []
                

                gnn_exp_model_input_explanation = []
                gnn_exp_model_input_explanation_each_pair_all_rep = []

                result = []
                topk_heap_min_score = 0.0
                all_rep_one_match_pair_num = 0
                
                for rep_idx, rep in enumerate(gen_rep_list):
    
                    if enable_topk == 0:
                        result = makex.sale_predictor.ScoreFunction_U_V(rep, graph, user_id, item_id, True, rep_idx, enable_topk, topk_explanation_scores_list, topk)
                        match_flag = result[5]
                        if match_flag != -1:
                            all_rep_one_match_pair_num += 1
                            all_rep_all_match_pair_num += 1
                            sorted_topk_explanation_rep_id.append(rep_idx)


                    if enable_topk == 1:
                        if topk_explanation:
                            topk_explanation_scores_list = [item[0] for item in topk_explanation]
                        else:
                            topk_explanation_scores_list = []
                        if len(topk_explanation) > 0:
                            topk_heap_min_score = topk_explanation[-1][0]
                        if rep.score < topk_heap_min_score:
                            continue
                        result = makex.sale_predictor.ScoreFunction_U_V(rep, graph, user_id, item_id, True, rep_idx, enable_topk, topk_explanation_scores_list, topk)

                        match_flag = result[1]
                        if match_flag != -1:
                            all_rep_one_match_pair_num += 1
                            all_rep_all_match_pair_num += 1
                            topk_explanation_rep_id_has_match.append(rep_idx)

                        return_explanation_result = result[0]

                        for updated_idx, explanation in enumerate(return_explanation_result):
                            if len(explanation[1]) > 0:
                                topk_explanation_rep_id.append(rep_idx)
                                topk_explanation.append(explanation)

                                gnn_exp_model_input_explanation_list = []
                                gnn_exp_model_input_explanation_list.append(explanation)
                                gnn_exp_model_input_explanation_list.append(rep.GetREPMatchArg())
                                gnn_exp_model_input_explanation_list.append(rep_idx)

                                gnn_exp_model_input_explanation.append(gnn_exp_model_input_explanation_list)
                                gnn_exp_model_input_explanation_each_pair_all_rep.append(gnn_exp_model_input_explanation_list)
                        combined_list = list(zip(topk_explanation, topk_explanation_rep_id))
                        if combined_list:
                            sorted_combined_list = sorted(combined_list, key=lambda x: x[0][0], reverse=True)
                            sorted_topk_explanation, sorted_topk_explanation_rep_id_tuple = zip(*sorted_combined_list)
                            topk_explanation = list(sorted_topk_explanation)[:topk]
                            sorted_topk_explanation_rep_id = list(sorted_topk_explanation_rep_id_tuple)[:topk]
                        
                        topk_explanation = sorted(topk_explanation, key=lambda x: x[0], reverse=True)[:topk]
                        gnn_exp_model_input_explanation = sorted(gnn_exp_model_input_explanation, key=lambda x: x[0][0], reverse=True)[:topk]
                        
                        gnn_exp_model_input_explanation_each_pair_all_rep = sorted(gnn_exp_model_input_explanation_each_pair_all_rep, key=lambda x: x[0][0], reverse=True)


                for topk_index, topk_rep_id in enumerate(sorted_topk_explanation_rep_id):
                    row_edge = []
                    row_edge.append(int(pair_id))
                    row_edge.append(int(user_id))
                    row_edge.append(int(item_id))
                    row_edge.append(int(topk_index))
                    row_edge.append(int(topk_rep_id))

                    with open(topk_rep_id_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row_edge)

                makex_explanation_v = args.makex_explanation_v
                makex_explanation_e = args.makex_explanation_e

                explanation_rep_to_csv(gnn_exp_model_input_explanation,pair_id,df_original_v, makex_explanation_v, makex_explanation_e)
                
                
                pair_id +=1

                topk_explanation = None
                topk_explanation_scores_list = None
                topk_explanation_rep_id = None

                gnn_exp_model_input_explanation = None
                gnn_exp_model_input_explanation_each_pair_all_rep = None
                result = None

if __name__ == "__main__":
    main(parse_args())
