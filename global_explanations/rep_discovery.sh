#!/bin/bash


pattern_file="../DataSets/Movielens/pattern.txt"
candidate_predicates_file="../DataSets/Movielens/candidate_predicates.txt"
rep_support=100000
rep_conf=0.7
rep_to_path_ratio=0.2
each_node_predicates=3
sort_by_support_weights=0.3
v_file="../DataSets/Movielens/original_graph/movielens_v.csv"
e_file="../DataSets/Movielens/original_graph/movielens_e.csv"
ml_file="../DataSets/Movielens/train_test/train.log"
delta_l=0.0
delta_r=1.0
user_offset=0
rep_file_generate="./rep_all.txt"
rep_file_generate_support_conf="./rep_support_conf.txt"
edge_label_reverse_csv="../DataSets/Movielens/edge_label_reverse.csv"
rep_file_generate_support_conf_none_support="./rep.txt"
num_process=25

g++ makex_rep_discovery.cpp -o makex_rep_discovery -std=c++17 -I ../pyMakex/include/ -fopenmp -O3 >> makex_rep_discovery.txt 2>&1


./makex_rep_discovery "$pattern_file" "$candidate_predicates_file" $rep_support $rep_conf $rep_to_path_ratio $each_node_predicates $sort_by_support_weights "$v_file" "$e_file" "$ml_file" $delta_l $delta_r $user_offset "$rep_file_generate" "$rep_file_generate_support_conf" "$edge_label_reverse_csv" "$rep_file_generate_support_conf_none_support" $num_process > ./output.txt 2>&1