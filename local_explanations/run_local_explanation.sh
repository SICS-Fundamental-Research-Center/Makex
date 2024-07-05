#!/bin/bash


pattern_nums=(50)
conf_limits=(0.7)
supp_limits=(150000)
each_pattern_rep_nums=(5)
random_seeds=(1996)
rep_num_ratios=(1.0)
vary_topk=(1 5 10 15)


model="hgt"
output_dir="./output//${model}/"
reserve_rep_dir="./reserve_rep/${model}/"

explanation_dir="./${model}/local_explanation/"


for pattern_num in "${pattern_nums[@]}"
do
  for conf_limit in "${conf_limits[@]}"
  do
    for supp_limit in "${supp_limits[@]}"
    do
      for each_pattern_rep_num in "${each_pattern_rep_nums[@]}"
      do
        for rep_num_ratio in "${rep_num_ratios[@]}"
        do
          for random_seed in "${random_seeds[@]}"
          do
            for topk in "${vary_topk[@]}"
            do
              rep_file="./rep.txt"
              makex_explanation_v="${explanation_dir}${topk}/topk${topk}/v.csv"
              makex_explanation_e="${explanation_dir}${topk}/N${topk}/e.csv"
              topk_rep_id_file="${explanation_dir}${topk}/N${topk}/topk_rep_id.csv"
              reserve_rep_file="${reserve_rep_dir}${model}_N${topk}.txt"
              output_file="${output_dir}${model}_N${topk}.csv"
              output_file_txt="${output_dir}${model}_N${topk}.txt"
              
              
              echo "Parameters passed to script: "
              echo "topk: $topk"
              exho "rep_file: $rep_file"
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
              
              
              ./local_explanation.sh "$pattern_num" "$conf_limit" "$supp_limit" "$each_pattern_rep_num" "$rep_file" "$rep_num_ratio" "$topk_rep_id_file" "$makex_explanation_v" "$makex_explanation_e" "$topk" "$random_seed" "$output_file" "$reserve_rep_file" >> "$output_file_txt" 2>&1
            done
          done
        done
      done
    done
  done
done
