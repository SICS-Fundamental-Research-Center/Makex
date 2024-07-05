import pandas as pd
import numpy as np
import csv
import os

def fidelity_factual_precision(df, k):
    
    user_items = df['rec'].to_numpy()
    print(len(user_items))

    sum = 0
    for i in range(len(user_items)):
         sum = sum + user_items[i]

    print(sum)
    precision = (1 / len(user_items)) * sum
    precision2 = (1 / 1000) * sum
    print("precision@{}:".format(k), precision, precision2)

    return precision, precision2


print("the dataset...")
file_location = "../../../dataset"
print('Enter dataset name:')
dataset_name = input()
print('Enter gnn model name:')
modelname = input()

res_filename = 'pge_result/' + modelname + '/'
dir_to_load = file_location + "/" + dataset_name + "/" 

up_dir_to_load = file_location + "/" + dataset_name + "/" + res_filename


eval_filename ='/eval.csv'

eval_df = pd.read_csv(up_dir_to_load + eval_filename, names= ['pair_id','topk','pred_score', 'rec', 'sparsity'])

max_k = 10

fid_columns = ['fid1','fid2']
fid_filename = "/fidelity_pre.csv"
with open(up_dir_to_load + fid_filename, 'w') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fid_columns)
    cur_fid, cur_fid2 = fidelity_factual_precision(eval_df, max_k)

    temp_dict = {
            'fid1': cur_fid, 
            'fid2': cur_fid2
    }
    writer.writerow(temp_dict)
    print("Fidelity computation finished...")
