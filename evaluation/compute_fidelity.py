import pandas as pd
import numpy as np
import csv
import os

def fidelity_factual_precision(df, k):
    
    max_pair_id = df['pair_id'].max()
    print("this pair has {} pairs".format(max_pair_id))
    user_items = df['rec'].to_numpy()
    num_pairs_used = int(min(k,len(user_items)))
    used_uis = user_items[:num_pairs_used]
    print(len(used_uis))

    sum = 0
    for i in range(len(used_uis)):
         sum = sum + used_uis[i]

    print(sum)
    precision = (1 / max_pair_id) * sum if max_pair_id > len(used_uis) else (1 / len(used_uis)) * sum
    precision2 = (1 / 1000) * sum
    print("precision@{}:".format(k), precision, precision2)

    return precision, precision2



print("the dataset...")
file_location = "../../../dataset"
print('Enter dataset name:')
dataset_name = input()
print('Enter the exact folder name:')
res_dir = input()
print('Enter gnn model name:')
modelname = input()

res_filename =  'makex_result/' + modelname + '/' + res_dir + '/'
dir_to_load = file_location + "/" + dataset_name + "/" 

up_dir_to_load = file_location + "/" + dataset_name + "/" + res_filename

sub_dirs = os.listdir(up_dir_to_load)

total_res = {}
for sub_dir in sub_dirs:
    resdir_to_load = up_dir_to_load + sub_dir + '/'
    print(resdir_to_load)
    if os.path.isdir(resdir_to_load) == False:
        continue
    eval_filename ='/eval.csv'
    eval_df = pd.read_csv(resdir_to_load + eval_filename, names= ['pair_id','topk','pred_score', 'rec'])
    pairs_range = {1000}

    fid_columns = ['fid1','fid2']
    fid_filename = "/fidelity_pre.csv"
    with open(resdir_to_load + fid_filename, 'w') as outfile:
            for max_k in pairs_range:
                 print(max_k)
                 writer = csv.DictWriter(outfile, fieldnames=fid_columns)
                 cur_fid, cur_fid2 = fidelity_factual_precision(eval_df, max_k)
                 temp_dict = {
                        'fid1': cur_fid, 
                        'fid2': cur_fid2
                  }
                 writer.writerow(temp_dict)
                 print("Fidelity computation finished...")
