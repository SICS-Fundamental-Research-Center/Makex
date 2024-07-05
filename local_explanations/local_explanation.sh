#!/bin/bash

pattern_num=$1
conf_limit=$2
supp_limit=$3
each_pattern_rep_num=$4
rep_file=$5
rep_num_ratio=$6
topk_rep_id_file=$7
makex_explanation_v=$8
makex_explanation_e=$9
topk=${10}
random_seed=${11}
output_file=${12}
reserve_rep_file=${13}
output_file_txt=${14}


echo "pattern_num: $pattern_num"
echo "conf_limit: $conf_limit"
echo "supp_limit: $supp_limit"
echo "each_pattern_rep_num: $each_pattern_rep_num"
echo "rep_file: $rep_file"
echo "rep_num_ratio: $rep_num_ratio"
echo "topk_rep_id_file: $topk_rep_id_file"
echo "makex_explanation_v: $makex_explanation_v"
echo "makex_explanation_e: $makex_explanation_e"
echo "random_seed: $random_seed"
echo "output_file: $output_file"
echo "reserve_rep_file: $reserve_rep_file"
echo "output_file_txt: $output_file_txt"

python ./local_explanation.py \
--pattern_num "$pattern_num" \
--conf_limit "$conf_limit" \
--supp_limit "$supp_limit" \
--each_pattern_rep "$each_pattern_rep_num" \
--rep_file "$rep_file" \
--rep_num_ratio "$rep_num_ratio" \
--topk_rep_id_file "$topk_rep_id_file" \
--makex_explanation_v "$makex_explanation_v" \
--makex_explanation_e "$makex_explanation_e" \
--topk "$topk" \
--random_seed "$random_seed" \
--output_file "$output_file" \
--reserve_rep_file "$reserve_rep_file" \
--edge_label_reverse_csv ../DataSets/Movielens/edge_label_reverse.csv \
--v_path ../DataSets/Movielens/original_graph/movielens_v.csv \
--e_path ../DataSets/Movielens/original_graph/movielens_e.csv \
--subgraph_path ../DataSets/Movielens/subgraph_by_subgraphx/hgt/explain_sgs_largeuser.csv \
--subgraph_pivot_path ../DataSets/Movielens/subgraph_by_subgraphx/hgt/explain_sgs_prop_largeuser.csv \ 
--ml_path ../DataSets/Movielens/train_test/train.log \
--delta_l 0.0  --delta_r 1.0 \
--tie_num 100 \
--sort_criteria conf \
--has_ml 1 \
--max_degree 3 --max_length 2 --max_subgraph_to_pattern_num 2 \
--hop_decay_factor 0.8 \
--enable_topk 1 \
--sample_pair_num 1000 \
--test_sample_pairs_file ../DataSets/Movielens/test_sample_pairs/hgt_sample_pairs.csv \