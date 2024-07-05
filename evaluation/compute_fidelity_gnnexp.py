import pandas as pd
import numpy as np
import csv
import os

def fidelity_factual_precision(df, k):
    
    grouped = df.groupby(['pair_id'])
    def compute_fid_u_v(user_group):
        user_items = user_group['rec'].tolist()
        return sum([user_items[i] for i in range(len(user_items))])

    grouped_fid_u_v = grouped.apply(compute_fid_u_v)
    precision = (1 / len(grouped_fid_u_v)) * sum(grouped_fid_u_v )
    precision2 = (1 / 1000) * sum(grouped_fid_u_v )
    print("precision@{}:".format(k), precision, precision2)

    return precision, precision2

def fidelity_cf_precision(df, k):
    
    grouped = df.groupby(['pair_id'])
    def compute_fid_u_v(user_group):
        user_items = user_group['rec'].tolist()
        return sum([user_items[i] for i in range(len(user_items))])

    grouped_fid_u_v = grouped.apply(compute_fid_u_v)
    precision = (1 / len(grouped_fid_u_v)) * (len(grouped_fid_u_v) - sum(grouped_fid_u_v ))
    precision2 = (1 / 1000) * (1000 - sum(grouped_fid_u_v ))
    print("precision@{}:".format(k), precision, precision2)

    return precision, precision2

print("the dataset...")
file_location = "../../../dataset"
dataset_name = "ciao"
print('Enter gnn model name:')
modelname = input()

factual_flag = False
res_filename = 'gnnexp_result/' + modelname + '/'  if factual_flag else 'cfexp_result/' + modelname + '/' 
dir_to_load = file_location + "/" + dataset_name + "/" 

up_dir_to_load = file_location + "/" + dataset_name + "/" + res_filename

sub_dirs = os.listdir(up_dir_to_load)


eval_filename ='/eval.csv'

eval_df = pd.read_csv(up_dir_to_load + eval_filename, names= ['pair_id','topk','pred_score', 'rec', 'total_rm_num', 'sparsity'])

max_k = 10

fid_columns = ['fid1','fid2']
fid_filename = "/fidelity_pre.csv"
with open(up_dir_to_load + fid_filename, 'w') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fid_columns)
    if factual_flag:
        cur_fid, cur_fid2 = fidelity_factual_precision(eval_df, max_k)
        temp_dict = {
                'fid1': cur_fid, 
                'fid2': cur_fid2
        }
        writer.writerow(temp_dict)
        print("Fidelity computation finished...")
    else:
        cur_fid, cur_fid2 = fidelity_cf_precision(eval_df, max_k)
        temp_dict = {
                'fid1': cur_fid, 
                'fid2': cur_fid2
        }
        writer.writerow(temp_dict)
        print("Fidelity computation finished...")